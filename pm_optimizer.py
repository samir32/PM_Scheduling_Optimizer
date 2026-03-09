"""
PM Schedule Optimizer — Streamlit Edition
Genetic algorithm for minimising peak weekly PM workload.

Fixes vs original tkinter version:
  - evaluate() fully vectorised (no Python loop over population)
  - mutate stub replaced with complete mutate_plans()
  - crossover fills odd-pop-size slots correctly (no zero rows)
  - fallback start logic actually relaxes restrictions
  - forbidden week selector covers any num_weeks (not just multiples of 52)
  - duplicate parameter parsing removed
  - progress shown in-browser via st.progress
  - results downloadable directly from the page
  - fitness convergence chart added
"""

import io
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch

# ─────────────────────────────────────────────────────────────────────────────
# Core GA Functions
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_vectorized(
    solutions, plan_list, plan_intervals, plan_work,
    num_weeks, device, forbidden_weeks_tensor, restriction_weight,
):
    """
    Fully-vectorised fitness evaluation.

    For every (plan, operation) pair the occurrence schedule is built as a
    (pop_size × max_occ) tensor in one shot — no Python loop over individuals.
    """
    pop_size = solutions.size(0)
    workload = torch.zeros(pop_size, num_weeks, device=device)
    flat_wl = workload.view(-1)                          # shared view
    row_idx = torch.arange(pop_size, device=device)

    for j, plan_name in enumerate(plan_list):
        starts = solutions[:, j] - 1                    # 0-indexed, (pop_size,)
        intervals_for_plan = plan_intervals[plan_name]
        work_for_plan = plan_work[plan_name]

        for i_op in range(len(intervals_for_plan)):
            step = intervals_for_plan[i_op].item()
            if step <= 0:
                continue
            wh = work_for_plan[i_op].item()

            min_start = starts.min().item()
            max_occ = int(np.ceil((num_weeks - min_start) / step)) + 1

            # occ[p, k] = start_p + k * step   shape: (pop_size, max_occ)
            k = torch.arange(0, max_occ, device=device)
            occ = starts.unsqueeze(1) + k.unsqueeze(0) * step
            valid_mask = (occ >= 0) & (occ < num_weeks)

            # Map to flat index in workload (pop_size * num_weeks)
            p_idx = row_idx.unsqueeze(1).expand_as(occ)
            flat_idx = (p_idx * num_weeks + occ.clamp(0, num_weeks - 1))[valid_mask]
            flat_wl.scatter_add_(
                0, flat_idx,
                torch.full((flat_idx.numel(),), wh, device=device),
            )

    workload = flat_wl.view(pop_size, num_weeks)

    # Apply penalty on forbidden weeks
    weighted = workload.clone()
    if forbidden_weeks_tensor.numel() > 0 and restriction_weight != 1.0:
        weighted[:, forbidden_weeks_tensor] *= restriction_weight

    max_work = weighted.max(dim=1).values
    min_work = weighted.min(dim=1).values
    # Lexicographic: first minimise peak, then minimise spread
    fitness = max_work * 10_000 + (max_work - min_work)
    return fitness


def select_elite(solutions, fitness, elite_count):
    return solutions[torch.argsort(fitness)[:elite_count]]


def crossover(parents, population_size, crossover_rate, device):
    """
    Fill new population. Pre-populate ALL slots with random parents first,
    so odd-sized populations never contain zero rows.
    """
    elite_count = parents.size(0)
    num_cols = parents.size(1)

    rand_idxs = torch.randint(0, elite_count, (population_size,), device=device)
    new_solutions = parents[rand_idxs].clone()   # every slot gets a real parent
    new_solutions[:elite_count] = parents         # elites survive unchanged

    for i in range(elite_count, population_size - 1, 2):
        if torch.rand(1, device=device) < crossover_rate:
            p1 = parents[torch.randint(0, elite_count, (1,), device=device)].squeeze(0)
            p2 = parents[torch.randint(0, elite_count, (1,), device=device)].squeeze(0)
            mask = torch.rand(num_cols, device=device) < 0.5
            new_solutions[i]     = torch.where(mask, p1, p2)
            new_solutions[i + 1] = torch.where(mask, p2, p1)
    return new_solutions


def mutate_plans(solutions, plan_list, allowed_starts_tensors, mutation_rate, device):
    """Randomly reassign start weeks to mutated individuals."""
    mut_mask = torch.rand_like(solutions.float()) < mutation_rate
    for j, plan_name in enumerate(plan_list):
        t = allowed_starts_tensors[plan_name]
        indices = mut_mask[:, j].nonzero(as_tuple=False).squeeze(1)
        if indices.numel() > 0:
            new_vals = t[torch.randint(0, len(t), (indices.numel(),), device=device)]
            solutions[indices, j] = new_vals
    return solutions


def compute_allowed_starts(
    plan_names, plan_intervals, num_weeks,
    use_restrictions, restricted_intervals,
    forbidden_weeks_tensor, device,
):
    """
    Return a dict {plan_name: [valid start weeks]}.
    If no feasible starts exist under restrictions, fall back to all candidates
    (proper fallback — previous code never actually relaxed restrictions).
    """
    allowed = {}
    max_neg = 4  # allow starting up to 4 weeks before week 1

    for plan_name in plan_names:
        ivals = plan_intervals[plan_name]
        max_start = min(int(ivals.min().item()), num_weeks)
        candidates = list(range(1 - max_neg, max_start + 1))
        good_starts = []

        for s in candidates:
            ok = True
            for step_t in ivals:
                step = step_t.item()
                occ = torch.arange(s, num_weeks, step, device=device).long()
                valid_occ = occ[occ >= 0]
                if (
                    use_restrictions
                    and step in restricted_intervals
                    and forbidden_weeks_tensor.numel() > 0
                    and valid_occ.numel() > 0
                    and (valid_occ.unsqueeze(1) == forbidden_weeks_tensor).any()
                ):
                    ok = False
                    break
            if ok:
                good_starts.append(s)

        # Proper fallback: accept all candidates without restriction check
        allowed[plan_name] = good_starts if good_starts else candidates

    return allowed


def run_ga(
    df, plan_col, interval_col, work_col,
    num_weeks, population_size, generations,
    elite_fraction, mutation_rate, crossover_rate,
    use_restrictions, restricted_intervals, forbidden_weeks,
    restriction_weight, device_str,
    progress_bar, status_text,
):
    device = torch.device(device_str)
    df = df.copy()
    df[plan_col] = df[plan_col].astype(str)

    plans = df.groupby(plan_col)
    plan_names = list(plans.groups.keys())
    num_plans = len(plan_names)
    elite_count = max(2, int(population_size * elite_fraction))

    plan_intervals, plan_work = {}, {}
    for name, group in plans:
        plan_intervals[name] = torch.tensor(
            group[interval_col].values.astype(float), dtype=torch.long, device=device
        )
        plan_work[name] = torch.tensor(
            group[work_col].values.astype(float), dtype=torch.float32, device=device
        )

    # Forbidden weeks tensor (0-indexed)
    if use_restrictions and forbidden_weeks:
        fw_tensor = torch.tensor(
            [w - 1 for w in forbidden_weeks if 0 <= w - 1 < num_weeks],
            dtype=torch.long, device=device,
        )
    else:
        fw_tensor = torch.tensor([], dtype=torch.long, device=device)

    allowed_starts = compute_allowed_starts(
        plan_names, plan_intervals, num_weeks,
        use_restrictions, restricted_intervals, fw_tensor, device,
    )
    allowed_tensors = {
        name: torch.tensor(starts, dtype=torch.long, device=device)
        for name, starts in allowed_starts.items()
    }

    # Initialise population
    solutions = torch.zeros(population_size, num_plans, dtype=torch.long, device=device)
    for j, name in enumerate(plan_names):
        t = allowed_tensors[name]
        solutions[:, j] = t[torch.randint(0, len(t), (population_size,), device=device)]

    best_solution, best_fitness_val = None, float("inf")
    fitness_history = []

    for gen in range(generations):
        fitness = evaluate_vectorized(
            solutions, plan_names, plan_intervals, plan_work,
            num_weeks, device, fw_tensor, restriction_weight,
        )

        min_f = fitness.min().item()
        fitness_history.append(min_f)

        if min_f < best_fitness_val:
            best_fitness_val = min_f
            best_solution = solutions[torch.argmin(fitness)].clone()

        # Live progress update
        progress_bar.progress((gen + 1) / generations)
        status_text.text(
            f"Generation {gen + 1}/{generations}  |  Best fitness: {best_fitness_val:,.1f}"
        )

        elite = select_elite(solutions, fitness, elite_count)
        solutions = crossover(elite, population_size, crossover_rate, device)
        solutions = mutate_plans(
            solutions, plan_names, allowed_tensors, mutation_rate, device
        )

    # Reconstruct final workload from best solution
    final_workload = torch.zeros(num_weeks, device=device)
    for j, name in enumerate(plan_names):
        start = best_solution[j].item()
        for i_op in range(len(plan_intervals[name])):
            step = plan_intervals[name][i_op].item()
            wh = plan_work[name][i_op].item()
            occ = torch.arange(start, num_weeks, step, device=device)
            valid_occ = occ[occ >= 0]
            if valid_occ.numel() > 0:
                final_workload.index_add_(
                    0, valid_occ,
                    torch.full((valid_occ.numel(),), wh, device=device),
                )

    final_workload_np = final_workload.cpu().numpy()
    best_sol_np = best_solution.cpu().numpy()
    plan_start_map = {plan_names[i]: int(best_sol_np[i]) for i in range(num_plans)}
    df["Start week"] = df[plan_col].map(plan_start_map)

    return df, final_workload_np, best_fitness_val, fitness_history


# ─────────────────────────────────────────────────────────────────────────────
# Streamlit App
# ─────────────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="PM Optimizer", page_icon="⚙️", layout="wide"
    )

    st.title("⚙️ PM Schedule Optimizer")
    st.caption(
        "Genetic algorithm to optimise preventive maintenance start weeks "
        "and minimise peak weekly workload."
    )

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Configuration")

        # Hardware
        with st.expander("Hardware", expanded=True):
            device_choice = st.selectbox("Device", ["CPU", "GPU (CUDA)"])
            use_gpu = device_choice == "GPU (CUDA)" and torch.cuda.is_available()
            if device_choice == "GPU (CUDA)" and not torch.cuda.is_available():
                st.warning("CUDA not available — using CPU.")
            device_str = "cuda" if use_gpu else "cpu"
            st.caption(f"Running on: **{device_str.upper()}**")

        # GA parameters
        with st.expander("GA Parameters", expanded=True):
            num_weeks   = st.number_input("Number of Weeks",  1,   520,  52,   step=1)
            pop_size    = st.number_input("Population Size",  100, 10000, 2000, step=100)
            generations = st.number_input("Generations",      10,  10000, 500,  step=10)
            elite_frac  = st.slider("Elite Fraction",  0.01, 0.50, 0.10, step=0.01,
                                    help="Fraction of best individuals carried to next generation")
            mut_rate    = st.slider("Mutation Rate",   0.001, 0.50, 0.05, step=0.005,
                                    help="Probability of randomly reassigning a plan's start week")
            cross_rate  = st.slider("Crossover Rate",  0.0, 1.0, 0.70, step=0.05,
                                    help="Probability of producing children by mixing two parents")

        # Restrictions toggle
        with st.expander("Restrictions", expanded=False):
            use_restrictions = st.checkbox("Enable Restrictions", value=False)
            restriction_weight = st.number_input(
                "Forbidden Week Penalty Weight",
                min_value=1.0, max_value=100.0, value=5.0, step=0.5,
                disabled=not use_restrictions,
                help="Multiplier applied to workload in forbidden weeks during fitness calculation",
            )

    # ── Step 1: File Upload ───────────────────────────────────────────────────
    st.header("Step 1 — Load Data")
    uploaded = st.file_uploader("Upload Excel file (.xlsx / .xls)", type=["xlsx", "xls"])

    df = None
    selected_sheet = None

    if uploaded:
        try:
            xls = pd.ExcelFile(uploaded)
            sheet_names = xls.sheet_names
        except Exception as e:
            st.error(f"Could not read file: {e}")
            sheet_names = []

        if sheet_names:
            selected_sheet = st.selectbox("Sheet", sheet_names)
            try:
                df = pd.read_excel(uploaded, sheet_name=selected_sheet)
                # Clean column names
                df.columns = [str(c) for c in df.columns]
                df = df[[c for c in df.columns if c and c.lower() != "nan"]]
                st.success(f"Loaded **{len(df):,} rows × {len(df.columns)} columns** from *{selected_sheet}*")
                with st.expander("Preview (first 20 rows)"):
                    st.dataframe(df.head(20), use_container_width=True)
            except Exception as e:
                st.error(f"Failed to load sheet: {e}")
                df = None

    # ── Step 2: Column Mapping ────────────────────────────────────────────────
    col_ok = False
    plan_col = interval_col = work_col = operation_col = None

    if df is not None:
        st.header("Step 2 — Map Columns")
        cols = list(df.columns)
        c1, c2, c3, c4 = st.columns(4)
        plan_col      = c1.selectbox("Plan",      cols, help="Groups operations into a maintenance plan")
        interval_col  = c2.selectbox("Interval",  cols, help="Recurrence interval in weeks")
        work_col      = c3.selectbox("Work (h)",  cols, help="Work-hours per occurrence")
        operation_col = c4.selectbox("Operation", cols, help="Operation identifier (informational)")

        # Validate numeric columns
        errors = []
        for col, label in [(interval_col, "Interval"), (work_col, "Work")]:
            try:
                pd.to_numeric(df[col], errors="raise")
            except Exception:
                errors.append(f"Column **{col}** ({label}) must be numeric.")
        for err in errors:
            st.error(err)
        col_ok = len(errors) == 0

    # ── Step 3: Restrictions (conditional) ───────────────────────────────────
    restricted_intervals: list = []
    forbidden_weeks: list = []

    if df is not None and use_restrictions:
        st.header("Step 3 — Restrictions")
        rc1, rc2 = st.columns([1, 2])

        with rc1:
            st.subheader("Intervals to restrict")
            if interval_col:
                try:
                    unique_ivals = sorted(
                        df[interval_col].dropna().astype(int).unique().tolist()
                    )
                    restricted_intervals = st.multiselect(
                        "Select interval values that must avoid forbidden weeks:",
                        options=unique_ivals,
                    )
                except Exception:
                    st.warning("Could not parse interval column values.")

        with rc2:
            st.subheader("Forbidden weeks")
            all_weeks = list(range(1, int(num_weeks) + 1))
            forbidden_weeks = st.multiselect(
                "Select weeks where PM should not be scheduled:",
                options=all_weeks,
            )
            if forbidden_weeks:
                # Quick visual summary
                fw_arr = np.zeros(int(num_weeks))
                for w in forbidden_weeks:
                    fw_arr[w - 1] = 1
                fig_fw = go.Figure(
                    go.Bar(
                        x=all_weeks, y=fw_arr,
                        marker_color=["#e74c3c" if w in forbidden_weeks else "#ecf0f1"
                                      for w in all_weeks],
                        showlegend=False,
                        hovertemplate="Week %{x}<extra></extra>",
                    )
                )
                fig_fw.update_layout(
                    height=140, margin=dict(l=0, r=0, t=10, b=0),
                    yaxis=dict(visible=False), xaxis_title="Week",
                    template="plotly_white",
                )
                st.plotly_chart(fig_fw, use_container_width=True)
                st.caption(f"{len(forbidden_weeks)} forbidden weeks selected")

    # ── Run Button ────────────────────────────────────────────────────────────
    run_header = "Step 4 — Run" if use_restrictions else "Step 3 — Run"
    st.header(run_header)

    if not (df is not None and col_ok):
        st.info("Complete the steps above to enable the optimizer.")

    run_clicked = st.button(
        "🚀 Run Optimization",
        disabled=not (df is not None and col_ok),
        type="primary",
        use_container_width=True,
    )

    if run_clicked:
        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            result_df, workload_np, best_fitness, fitness_history = run_ga(
                df=df,
                plan_col=plan_col,
                interval_col=interval_col,
                work_col=work_col,
                num_weeks=int(num_weeks),
                population_size=int(pop_size),
                generations=int(generations),
                elite_fraction=float(elite_frac),
                mutation_rate=float(mut_rate),
                crossover_rate=float(cross_rate),
                use_restrictions=use_restrictions,
                restricted_intervals=list(restricted_intervals),
                forbidden_weeks=list(forbidden_weeks),
                restriction_weight=float(restriction_weight),
                device_str=device_str,
                progress_bar=progress_bar,
                status_text=status_text,
            )
            # Persist results across reruns
            st.session_state["result_df"]      = result_df
            st.session_state["workload_np"]    = workload_np
            st.session_state["best_fitness"]   = best_fitness
            st.session_state["fitness_history"] = fitness_history
            st.session_state["num_weeks"]      = int(num_weeks)
            st.session_state["forbidden_weeks"] = list(forbidden_weeks)
            st.session_state["sheet_name"]     = selected_sheet or "Optimized Schedule"
        except Exception as e:
            st.error(f"Optimization failed: {e}")
            raise

    # ── Results ───────────────────────────────────────────────────────────────
    if "workload_np" in st.session_state:
        wl  = st.session_state["workload_np"]
        rdf = st.session_state["result_df"]
        n_w = st.session_state["num_weeks"]
        fw  = st.session_state["forbidden_weeks"]
        fh  = st.session_state["fitness_history"]
        sht = st.session_state["sheet_name"]

        st.success("✅ Optimization complete!")
        st.divider()

        # ── Key metrics ───────────────────────────────────────────────────────
        st.subheader("Summary")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Peak Workload",  f"{wl.max():.1f} h")
        m2.metric("Mean Workload",  f"{wl.mean():.1f} h")
        m3.metric("Min Workload",   f"{wl.min():.1f} h")
        m4.metric("Std Dev",        f"{wl.std():.1f} h")

        # ── Workload chart ────────────────────────────────────────────────────
        st.subheader("Workload Distribution by Week")
        week_labels = list(range(1, n_w + 1))
        bar_colors = [
            "#e74c3c" if w in fw else "#3498db" for w in week_labels
        ]

        fig_wl = go.Figure()
        fig_wl.add_trace(
            go.Bar(
                x=week_labels, y=wl.tolist(),
                marker_color=bar_colors,
                name="Workload",
                hovertemplate="Week %{x}<br><b>%{y:.1f} h</b><extra></extra>",
            )
        )
        fig_wl.add_hline(
            y=float(wl.mean()),
            line_dash="dash", line_color="orange", line_width=1.5,
            annotation_text=f"Mean: {wl.mean():.1f} h",
            annotation_position="top right",
        )
        if fw:
            # Phantom trace just for legend entry
            fig_wl.add_trace(
                go.Bar(
                    x=[None], y=[None],
                    marker_color="#e74c3c",
                    name="Forbidden week",
                )
            )
        fig_wl.update_layout(
            xaxis_title="Week",
            yaxis_title="Workload (Hours)",
            height=420,
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            bargap=0.1,
        )
        st.plotly_chart(fig_wl, use_container_width=True)

        # ── Convergence chart ─────────────────────────────────────────────────
        with st.expander("Fitness Convergence"):
            fig_cv = go.Figure()
            fig_cv.add_trace(
                go.Scatter(
                    x=list(range(1, len(fh) + 1)),
                    y=fh,
                    mode="lines",
                    name="Best Fitness",
                    line=dict(color="#2ecc71", width=2),
                    hovertemplate="Gen %{x}<br>Fitness: %{y:,.1f}<extra></extra>",
                )
            )
            fig_cv.update_layout(
                xaxis_title="Generation",
                yaxis_title="Best Fitness (lower = better)",
                height=320,
                template="plotly_white",
            )
            st.plotly_chart(fig_cv, use_container_width=True)

        # ── Results table ─────────────────────────────────────────────────────
        st.subheader("Optimized Schedule")
        st.dataframe(rdf, use_container_width=True, height=320)

        # ── Download ──────────────────────────────────────────────────────────
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            rdf.to_excel(writer, sheet_name=sht[:31], index=False)
        buf.seek(0)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        st.download_button(
            label="⬇️ Download Optimized Schedule (.xlsx)",
            data=buf,
            file_name=f"optimized_schedule_{timestamp}.xlsx",
            mime=(
                "application/vnd.openxmlformats-officedocument"
                ".spreadsheetml.sheet"
            ),
            use_container_width=True,
        )


if __name__ == "__main__":
    main()
