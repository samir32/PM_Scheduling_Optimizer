"""
PM Schedule Optimizer — Streamlit Edition
Run with:  streamlit run pm_optimizer.py
"""

import io
import logging
import sys
from datetime import datetime

for _name in [
    "streamlit.runtime.scriptrunner_utils.script_run_context",
    "streamlit.runtime.scriptrunner.script_run_context",
]:
    logging.getLogger(_name).setLevel(logging.ERROR)

_missing = []
for _pkg in ["streamlit", "pandas", "numpy", "torch", "plotly", "openpyxl"]:
    try:
        __import__(_pkg)
    except ImportError:
        _missing.append(_pkg)

if _missing:
    print("\n[ERROR] Missing packages:", ", ".join(_missing))
    print("Fix with:  pip install", " ".join(_missing))
    sys.exit(1)

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch


# ─────────────────────────────────────────────────────────────────────────────
# Unit conversion
# ─────────────────────────────────────────────────────────────────────────────

# Canonical aliases for each unit (case-insensitive)
UNIT_ALIASES = {
    "day":   ["day",   "days",   "d"],
    "week":  ["week",  "weeks",  "w", "wk",  "wks"],
    "month": ["month", "months", "m", "mo",  "mos"],
    "year":  ["year",  "years",  "y", "yr",  "yrs"],
}

def normalise_unit(raw) -> str:
    """Return 'week', 'month', or 'year'. Returns None if unrecognised."""
    s = str(raw).strip().lower()
    for canonical, aliases in UNIT_ALIASES.items():
        if s in aliases:
            return canonical
    return None

def interval_to_weeks(interval: float, unit: str,
                      days_per_week: float,
                      weeks_per_month: float,
                      weeks_per_year: float) -> float:
    """Convert an interval + unit into a float number of weeks."""
    if unit == "day":
        return float(interval) / days_per_week
    if unit == "week":
        return float(interval)
    if unit == "month":
        return float(interval) * weeks_per_month
    if unit == "year":
        return float(interval) * weeks_per_year
    raise ValueError(f"Unknown unit: {unit!r}")


import re as _re

# Patterns for embedded-unit strings
_DAYS_PAT  = _re.compile(r"^\s*(\d+(?:\.\d+)?)\s+days?\.?\s*$", _re.IGNORECASE)
_MINS_PAT  = _re.compile(r"^\s*(\d+(?:\.\d+)?)\s+min\.?\s*$",  _re.IGNORECASE)


def parse_interval_cell(val, fallback_unit: str):
    """
    Return (numeric_value, unit_str) from an interval cell.
    Handles embedded formats like "7 days" as well as plain numbers
    with a fallback unit from a separate column.
    """
    s = str(val).strip()
    m = _DAYS_PAT.match(s)
    if m:
        return float(m.group(1)), "day"
    # plain number — use fallback unit
    return float(val), fallback_unit


def parse_work_cell(val, global_work_unit: str):
    """
    Return work in HOURS from a work cell.
    Handles "55 min." strings; otherwise applies the global work unit toggle.
    """
    s = str(val).strip()
    m = _MINS_PAT.match(s)
    if m:
        return float(m.group(1)) / 60.0
    v = float(val)
    return v / 60.0 if global_work_unit == "Minutes" else v


# ─────────────────────────────────────────────────────────────────────────────
# Core GA functions
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_vectorized(solutions, plan_list, plan_intervals_weeks, plan_work,
                        num_weeks, device, forbidden_weeks_tensor, restriction_weight):
    """
    Fully-vectorised fitness evaluation.
    plan_intervals_weeks values are float tensors (weeks, possibly fractional).
    Occurrence indices are rounded to the nearest integer week.
    """
    pop_size = solutions.size(0)
    workload = torch.zeros(pop_size, num_weeks, device=device)
    flat_wl  = workload.view(-1)
    row_idx  = torch.arange(pop_size, device=device)

    for j, plan_name in enumerate(plan_list):
        starts = solutions[:, j].float() - 1.0       # 0-indexed, float

        for i_op in range(len(plan_intervals_weeks[plan_name])):
            step = plan_intervals_weeks[plan_name][i_op].item()   # float weeks
            if step <= 0:
                continue
            wh      = plan_work[plan_name][i_op].item()
            max_occ = int(np.ceil((num_weeks - starts.min().item()) / step)) + 1

            k        = torch.arange(0, max_occ, device=device, dtype=torch.float32)
            occ_f    = starts.unsqueeze(1) + k.unsqueeze(0) * step   # (pop, max_occ)
            occ      = occ_f.round().long()                           # round to nearest week
            valid    = (occ >= 0) & (occ < num_weeks)

            p_idx    = row_idx.unsqueeze(1).expand_as(occ)
            flat_idx = (p_idx * num_weeks + occ.clamp(0, num_weeks - 1))[valid]
            flat_wl.scatter_add_(0, flat_idx,
                                 torch.full((flat_idx.numel(),), wh, device=device))

    workload = flat_wl.view(pop_size, num_weeks)
    weighted = workload.clone()
    if forbidden_weeks_tensor.numel() > 0 and restriction_weight != 1.0:
        weighted[:, forbidden_weeks_tensor] *= restriction_weight

    max_w = weighted.max(dim=1).values
    min_w = weighted.min(dim=1).values
    return max_w * 10_000 + (max_w - min_w)


def select_elite(solutions, fitness, elite_count):
    return solutions[torch.argsort(fitness)[:elite_count]]


def crossover(parents, population_size, crossover_rate, device):
    elite_count   = parents.size(0)
    num_cols      = parents.size(1)
    rand_idxs     = torch.randint(0, elite_count, (population_size,), device=device)
    new_solutions = parents[rand_idxs].clone()
    new_solutions[:elite_count] = parents

    for i in range(elite_count, population_size - 1, 2):
        if torch.rand(1, device=device) < crossover_rate:
            p1   = parents[torch.randint(0, elite_count, (1,), device=device)].squeeze(0)
            p2   = parents[torch.randint(0, elite_count, (1,), device=device)].squeeze(0)
            mask = torch.rand(num_cols, device=device) < 0.5
            new_solutions[i]     = torch.where(mask, p1, p2)
            new_solutions[i + 1] = torch.where(mask, p2, p1)
    return new_solutions


def mutate_plans(solutions, plan_list, allowed_tensors, mutation_rate, device):
    mut_mask = torch.rand_like(solutions.float()) < mutation_rate
    for j, name in enumerate(plan_list):
        t       = allowed_tensors[name]
        indices = mut_mask[:, j].nonzero(as_tuple=False).squeeze(1)
        if indices.numel() > 0:
            new_vals = t[torch.randint(0, len(t), (indices.numel(),), device=device)]
            solutions[indices, j] = new_vals
    return solutions


def compute_allowed_starts(plan_names, plan_intervals_weeks, num_weeks,
                            use_restrictions, restricted_intervals_weeks,
                            forbidden_weeks_tensor, device):
    """
    restricted_intervals_weeks: set of float week values that must avoid
    forbidden weeks (after unit conversion).
    """
    allowed = {}
    for plan_name in plan_names:
        ivals      = plan_intervals_weeks[plan_name]   # float tensor
        min_step   = ivals.min().item()
        max_start  = min(int(np.ceil(min_step)), num_weeks)
        candidates = list(range(1 - 4, max_start + 1))
        good       = []

        for s in candidates:
            ok = True
            for step_t in ivals:
                step      = step_t.item()
                # Generate occurrences as floats, round to integer weeks
                occ_f     = torch.arange(s, num_weeks, step, device=device)
                valid_occ = occ_f[occ_f >= 0].round().long()
                valid_occ = valid_occ[valid_occ < num_weeks]

                if (use_restrictions
                        and any(abs(step - rs) < 0.01 for rs in restricted_intervals_weeks)
                        and forbidden_weeks_tensor.numel() > 0
                        and valid_occ.numel() > 0
                        and (valid_occ.unsqueeze(1) == forbidden_weeks_tensor).any()):
                    ok = False
                    break
            if ok:
                good.append(s)

        allowed[plan_name] = good if good else candidates
    return allowed


def run_ga(df, plan_col, interval_col, unit_col, work_col,
           work_unit, use_operation_col, operation_col,
           days_per_week, weeks_per_month, weeks_per_year,
           num_weeks, population_size, generations,
           elite_fraction, mutation_rate, crossover_rate,
           use_restrictions, restricted_interval_weeks, forbidden_weeks,
           restriction_weight, device_str,
           progress_bar, status_text):

    device       = torch.device(device_str)
    df           = df.copy()
    df[plan_col] = df[plan_col].astype(str)

    # Handle optional operation column
    if not use_operation_col or operation_col is None:
        df["_operation"] = 1
        operation_col = "_operation"

    # ── Parse intervals: handle embedded "7 days" strings or plain number + unit col ──
    def _parse_row_interval(row):
        fallback = normalise_unit(row[unit_col]) if unit_col else "week"
        if fallback is None:
            raise ValueError(
                f"Unrecognised unit '{row[unit_col]}' in row {row.name}. "
                f"Expected Day/Week/Month/Year."
            )
        numeric, unit = parse_interval_cell(row[interval_col], fallback)
        return interval_to_weeks(numeric, unit, days_per_week, weeks_per_month, weeks_per_year)

    df["_interval_weeks"] = df.apply(_parse_row_interval, axis=1)

    plans      = df.groupby(plan_col)
    plan_names = list(plans.groups.keys())
    num_plans  = len(plan_names)
    elite_count = max(2, int(population_size * elite_fraction))

    # Float tensors for intervals (weeks), float32 for work
    plan_intervals_weeks, plan_work = {}, {}
    for name, group in plans:
        plan_intervals_weeks[name] = torch.tensor(
            group["_interval_weeks"].values.astype(float),
            dtype=torch.float32, device=device)
        work_values = np.array([parse_work_cell(v, work_unit) for v in group[work_col].values],
                               dtype=float)
        plan_work[name] = torch.tensor(work_values, dtype=torch.float32, device=device)

    fw_tensor = (
        torch.tensor([w - 1 for w in forbidden_weeks if 0 <= w - 1 < num_weeks],
                     dtype=torch.long, device=device)
        if use_restrictions and forbidden_weeks
        else torch.tensor([], dtype=torch.long, device=device)
    )

    allowed_starts = compute_allowed_starts(
        plan_names, plan_intervals_weeks, num_weeks,
        use_restrictions, restricted_interval_weeks, fw_tensor, device)

    allowed_tensors = {
        name: torch.tensor(starts, dtype=torch.long, device=device)
        for name, starts in allowed_starts.items()
    }

    # Initialise population (start weeks are integers)
    solutions = torch.zeros(population_size, num_plans, dtype=torch.long, device=device)
    for j, name in enumerate(plan_names):
        t = allowed_tensors[name]
        solutions[:, j] = t[torch.randint(0, len(t), (population_size,), device=device)]

    best_solution, best_fitness_val = None, float("inf")
    fitness_history = []

    for gen in range(generations):
        fitness = evaluate_vectorized(
            solutions, plan_names, plan_intervals_weeks, plan_work,
            num_weeks, device, fw_tensor, restriction_weight)

        min_f = fitness.min().item()
        fitness_history.append(min_f)

        if min_f < best_fitness_val:
            best_fitness_val = min_f
            best_solution    = solutions[torch.argmin(fitness)].clone()

        progress_bar.progress((gen + 1) / generations)
        status_text.text(
            f"Generation {gen + 1}/{generations}  |  Best fitness: {best_fitness_val:,.1f}")

        elite     = select_elite(solutions, fitness, elite_count)
        solutions = crossover(elite, population_size, crossover_rate, device)
        solutions = mutate_plans(solutions, plan_names, allowed_tensors, mutation_rate, device)

    # ── Reconstruct final workload from best solution ─────────────────────────
    final_workload = torch.zeros(num_weeks, device=device)
    for j, name in enumerate(plan_names):
        start = best_solution[j].item()
        for i_op in range(len(plan_intervals_weeks[name])):
            step  = plan_intervals_weeks[name][i_op].item()
            wh    = plan_work[name][i_op].item()
            occ_f = torch.arange(start, num_weeks, step, device=device)
            occ   = occ_f[occ_f >= 0].round().long()
            occ   = occ[occ < num_weeks]
            if occ.numel() > 0:
                final_workload.index_add_(
                    0, occ, torch.full((occ.numel(),), wh, device=device))

    wl_np    = final_workload.cpu().numpy()
    best_np  = best_solution.cpu().numpy()
    plan_start_map = {plan_names[i]: int(best_np[i]) for i in range(num_plans)}
    df["Start week"] = df[plan_col].map(plan_start_map)
    df = df.drop(columns=["_unit_norm", "_interval_weeks", "_operation"], errors="ignore")

    return df, wl_np, best_fitness_val, fitness_history


# ─────────────────────────────────────────────────────────────────────────────
# Streamlit UI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(page_title="PM Optimizer", page_icon="⚙️", layout="wide")
    st.title("⚙️ PM Schedule Optimizer")
    st.caption(
        "Genetic algorithm to optimise preventive maintenance start weeks "
        "and minimise peak weekly workload."
    )

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("Configuration")

        device_str = "cpu"

        # GA parameters — always visible
        st.subheader("🧬 GA Parameters")
        num_weeks   = st.number_input("Number of Weeks",  min_value=1,   max_value=520,   value=52,   step=1)
        pop_size    = st.number_input("Population Size",  min_value=100, max_value=10000, value=2000, step=100)
        generations = st.number_input("Generations",      min_value=10,  max_value=5000,  value=500,  step=10)
        elite_frac  = st.slider("Elite Fraction",  min_value=0.01, max_value=0.50, value=0.10, step=0.01,
                                help="Fraction of best individuals carried unchanged to next generation")
        mut_rate    = st.slider("Mutation Rate",   min_value=0.001, max_value=0.50, value=0.05, step=0.005,
                                help="Probability of randomly reassigning a plan's start week")
        cross_rate  = st.slider("Crossover Rate",  min_value=0.0, max_value=1.0, value=0.70, step=0.05,
                                help="Probability of producing children by mixing two parents")

        st.divider()

        # Unit conversion factors
        st.subheader("📅 Unit Conversion")
        st.caption("Conversion factors for interval units.")
        days_per_week   = st.number_input("Days per Week",   min_value=1.0, max_value=7.0,
                                          value=7.0,   step=0.5)
        weeks_per_month = st.number_input("Weeks per Month", min_value=1.0, max_value=6.0,
                                          value=4.333, step=0.001, format="%.3f")
        weeks_per_year  = st.number_input("Weeks per Year",  min_value=48.0, max_value=56.0,
                                          value=52.0,  step=0.5)

        st.divider()
        st.subheader("🔧 Work Unit")
        work_unit = st.radio("Work values are in:", ["Hours", "Minutes"], horizontal=True,
                             help="Minutes will be divided by 60 — all outputs are in hours")

        st.divider()

        # Restrictions
        st.subheader("🔩 Operation Column")
        use_operation_col = st.checkbox("Use Operation Column", value=False,
                                         help="When disabled, all rows are assigned operation = 1")

        st.divider()
        st.subheader("🚫 Restrictions")
        use_restrictions   = st.checkbox("Enable Restrictions", value=False)
        restriction_weight = st.number_input(
            "Forbidden Week Penalty Weight",
            min_value=1.0, max_value=100.0, value=5.0, step=0.5,
            disabled=not use_restrictions,
            help="Multiplier applied to workload falling in forbidden weeks during fitness scoring")

    # ── Step 1 — Upload ───────────────────────────────────────────────────────
    st.header("Step 1 — Load Data")
    uploaded = st.file_uploader("Upload Excel file (.xlsx / .xls)", type=["xlsx", "xls"])

    df, selected_sheet = None, None

    if uploaded:
        try:
            xls         = pd.ExcelFile(uploaded)
            sheet_names = xls.sheet_names
        except Exception as e:
            st.error(f"Could not read file: {e}")
            sheet_names = []

        if sheet_names:
            selected_sheet = st.selectbox("Sheet", sheet_names)
            try:
                df = pd.read_excel(uploaded, sheet_name=selected_sheet)
                df.columns = [str(c) for c in df.columns]
                df = df[[c for c in df.columns if c.lower() != "nan"]]
                st.success(
                    f"Loaded **{len(df):,} rows × {len(df.columns)} columns** "
                    f"from *{selected_sheet}*")
                with st.expander("Preview (first 20 rows)"):
                    st.dataframe(df.head(20), use_container_width=True)
            except Exception as e:
                st.error(f"Failed to load sheet: {e}")
                df = None

    # ── Step 2 — Column Mapping ───────────────────────────────────────────────
    col_ok = False
    plan_col = interval_col = unit_col = work_col = operation_col = None
    unique_units = []

    # Default column names that trigger auto-selection
    DEFAULT_PLAN_COL      = "# Route"
    DEFAULT_INTERVAL_COL  = "Frequency (days)"
    DEFAULT_WORK_COL      = "totaltime_display"

    if df is not None:
        st.header("Step 2 — Map Columns")
        cols = list(df.columns)

        def _default_idx(preferred, cols):
            return cols.index(preferred) if preferred in cols else 0

        c1, c2, c3, c4, c5 = st.columns(5)
        plan_col      = c1.selectbox("# Route",             cols,
                                     index=_default_idx(DEFAULT_PLAN_COL, cols),
                                     help="Groups operations into a maintenance plan")
        interval_col  = c2.selectbox("Frequency (days)",    cols,
                                     index=_default_idx(DEFAULT_INTERVAL_COL, cols),
                                     help="Recurrence value. Plain number (uses Unit col) or embedded string like '7 days'")
        unit_col_opts = ["— none (embedded in Interval) —"] + cols
        unit_col_sel  = c3.selectbox("Unit (optional)", unit_col_opts,
                                     help="Unit column for plain-number intervals. Not needed if interval contains '7 days' etc.")
        unit_col      = None if unit_col_sel.startswith("—") else unit_col_sel
        work_col      = c4.selectbox("totaltime_display",   cols,
                                     index=_default_idx(DEFAULT_WORK_COL, cols),
                                     help="Work per occurrence. Plain number (uses sidebar unit) or '55 min.'")

        # Operation column — only shown when enabled in sidebar
        if use_operation_col:
            operation_col = c5.selectbox("Operation", cols,
                                          help="Operation identifier (informational)")
        else:
            c5.markdown("**Operation**")
            c5.caption("Disabled — rows will use value `1`")
            operation_col = None

        errors = []

        # ── Validate & preview interval column ────────────────────────────────
        try:
            sample_parsed = []
            for val in df[interval_col].dropna().unique()[:20]:
                fallback = "week"
                if unit_col:
                    # find the unit for this row by matching a sample row
                    sample_row = df[df[interval_col] == val].iloc[0]
                    raw_u = normalise_unit(sample_row[unit_col])
                    if raw_u is None:
                        errors.append(f"Unrecognised unit value `{sample_row[unit_col]}` in column **{unit_col}**.")
                        raise StopIteration
                    fallback = raw_u
                numeric, unit = parse_interval_cell(val, fallback)
                wks = interval_to_weeks(numeric, unit, days_per_week, weeks_per_month, weeks_per_year)
                sample_parsed.append(f"`{val}` → **{wks:.2f} wks**")
            if not errors:
                st.info("Interval preview (up to 20 unique values):  " + "  |  ".join(sample_parsed))
        except StopIteration:
            pass
        except Exception as e:
            errors.append(f"Could not parse Interval column: {e}")

        # ── Validate & preview work column ────────────────────────────────────
        try:
            work_sample = []
            for val in df[work_col].dropna().unique()[:10]:
                h = parse_work_cell(val, work_unit)
                work_sample.append(f"`{val}` → **{h:.3f} h**")
            if not errors:
                st.info("Work preview (up to 10 unique values):  " + "  |  ".join(work_sample))
        except Exception as e:
            errors.append(f"Could not parse Work column: {e}")

        for err in errors:
            st.error(err)
        col_ok = len(errors) == 0

    # ── Step 3 — Restrictions (optional) ─────────────────────────────────────
    restricted_interval_weeks = set()
    forbidden_weeks           = []

    if df is not None and col_ok and use_restrictions:
        st.header("Step 3 — Restrictions")
        rc1, rc2 = st.columns([1, 2])

        with rc1:
            st.subheader("Intervals to restrict")
            st.caption("Select which intervals must avoid forbidden weeks.")
            if interval_col:
                try:
                    cols_to_use = [interval_col] + ([unit_col] if unit_col else [])
                    interval_rows = df[cols_to_use].dropna().drop_duplicates().copy()

                    def _row_to_weeks(r):
                        fallback = normalise_unit(r[unit_col]) if unit_col else "week"
                        num, unit = parse_interval_cell(r[interval_col], fallback or "week")
                        return interval_to_weeks(num, unit, days_per_week, weeks_per_month, weeks_per_year)

                    interval_rows["_weeks"] = interval_rows.apply(_row_to_weeks, axis=1)
                    interval_rows["_label"] = interval_rows.apply(
                        lambda r: f"{r[interval_col]}  ({r['_weeks']:.2f} wks)", axis=1)
                    interval_rows = interval_rows.sort_values("_weeks")

                    label_to_weeks = dict(zip(interval_rows["_label"], interval_rows["_weeks"]))
                    selected_labels = st.multiselect(
                        "Intervals that must avoid forbidden weeks:",
                        options=list(label_to_weeks.keys()))
                    restricted_interval_weeks = {label_to_weeks[l] for l in selected_labels}

                except Exception as e:
                    st.warning(f"Could not build interval list: {e}")

        with rc2:
            st.subheader("Forbidden weeks")
            all_weeks       = list(range(1, int(num_weeks) + 1))
            forbidden_weeks = st.multiselect(
                "Weeks where PM should not be scheduled:", options=all_weeks)

            if forbidden_weeks:
                fw_arr  = [1 if w in set(forbidden_weeks) else 0 for w in all_weeks]
                fig_fw  = go.Figure(go.Bar(
                    x=all_weeks, y=fw_arr,
                    marker_color=["#e74c3c" if v else "#ecf0f1" for v in fw_arr],
                    hovertemplate="Week %{x}<extra></extra>", showlegend=False))
                fig_fw.update_layout(
                    height=130, margin=dict(l=0, r=0, t=10, b=0),
                    yaxis=dict(visible=False), xaxis_title="Week",
                    template="plotly_white")
                st.plotly_chart(fig_fw, use_container_width=True)
                st.caption(f"{len(forbidden_weeks)} forbidden weeks selected")

    # ── Run ───────────────────────────────────────────────────────────────────
    step_n = 4 if (use_restrictions and col_ok and df is not None) else 3
    st.header(f"Step {step_n} — Run")

    if not (df is not None and col_ok):
        st.info("Complete the steps above to enable the optimizer.")

    run_clicked = st.button(
        "🚀 Run Optimization",
        disabled=not (df is not None and col_ok),
        type="primary",
        use_container_width=True)

    if run_clicked:
        progress_bar = st.progress(0)
        status_text  = st.empty()
        try:
            result_df, wl_np, best_fitness, fh = run_ga(
                df=df,
                plan_col=plan_col, interval_col=interval_col,
                unit_col=unit_col, work_col=work_col,
                work_unit=work_unit,
                use_operation_col=use_operation_col,
                operation_col=operation_col,
                days_per_week=float(days_per_week),
                weeks_per_month=float(weeks_per_month),
                weeks_per_year=float(weeks_per_year),
                num_weeks=int(num_weeks),
                population_size=int(pop_size),
                generations=int(generations),
                elite_fraction=float(elite_frac),
                mutation_rate=float(mut_rate),
                crossover_rate=float(cross_rate),
                use_restrictions=use_restrictions,
                restricted_interval_weeks=restricted_interval_weeks,
                forbidden_weeks=list(forbidden_weeks),
                restriction_weight=float(restriction_weight),
                device_str=device_str,
                progress_bar=progress_bar,
                status_text=status_text,
            )
            st.session_state.update({
                "result_df":       result_df,
                "workload_np":     wl_np,
                "best_fitness":    best_fitness,
                "fitness_history": fh,
                "num_weeks":       int(num_weeks),
                "forbidden_weeks": list(forbidden_weeks),
                "sheet_name":      selected_sheet or "Schedule",
            })
        except Exception as e:
            st.error(f"Optimization error: {e}")
            st.exception(e)

    # ── Results ───────────────────────────────────────────────────────────────
    if "workload_np" in st.session_state:
        wl  = st.session_state["workload_np"]
        rdf = st.session_state["result_df"]
        n_w = st.session_state["num_weeks"]
        fw  = set(st.session_state["forbidden_weeks"])
        fh  = st.session_state["fitness_history"]
        sht = st.session_state["sheet_name"]

        st.success("✅ Optimization complete!")
        st.divider()

        # Metrics
        st.subheader("Summary")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Peak Workload", f"{wl.max():.1f} h")
        m2.metric("Mean Workload", f"{wl.mean():.1f} h")
        m3.metric("Min Workload",  f"{wl.min():.1f} h")
        m4.metric("Std Dev",       f"{wl.std():.1f} h")

        # Workload chart
        st.subheader("Workload Distribution by Week")
        weeks   = list(range(1, n_w + 1))
        fig_wl  = go.Figure()
        fig_wl.add_trace(go.Bar(
            x=weeks, y=wl.tolist(),
            marker_color=["#e74c3c" if w in fw else "#3498db" for w in weeks],
            hovertemplate="Week %{x}<br><b>%{y:.1f} h</b><extra></extra>",
            name="Workload"))
        fig_wl.add_hline(
            y=float(wl.mean()), line_dash="dash", line_color="orange",
            annotation_text=f"Mean: {wl.mean():.1f} h",
            annotation_position="top right")
        if fw:
            fig_wl.add_trace(go.Bar(x=[None], y=[None],
                                    marker_color="#e74c3c", name="Forbidden week"))
        fig_wl.update_layout(
            xaxis_title="Week", yaxis_title="Workload (Hours)",
            height=420, template="plotly_white", bargap=0.1,
            legend=dict(orientation="h", yanchor="bottom", y=1.02))
        st.plotly_chart(fig_wl, use_container_width=True)

        # Convergence
        with st.expander("Fitness Convergence"):
            fig_cv = go.Figure(go.Scatter(
                x=list(range(1, len(fh) + 1)), y=fh, mode="lines",
                line=dict(color="#2ecc71", width=2),
                hovertemplate="Gen %{x}<br>Fitness: %{y:,.1f}<extra></extra>"))
            fig_cv.update_layout(
                xaxis_title="Generation",
                yaxis_title="Best Fitness (lower = better)",
                height=300, template="plotly_white")
            st.plotly_chart(fig_cv, use_container_width=True)

        # Table
        st.subheader("Optimized Schedule")
        st.dataframe(rdf, use_container_width=True, height=320)

        # Download
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            rdf.to_excel(writer, sheet_name=sht[:31], index=False)
        buf.seek(0)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        st.download_button(
            label="⬇️ Download Optimized Schedule (.xlsx)",
            data=buf,
            file_name=f"optimized_schedule_{ts}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True)


if __name__ == "__main__":
    main()
