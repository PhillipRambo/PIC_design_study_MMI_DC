import matplotlib.pyplot as plt
import numpy as np
import tidy3d as td
import gdstk



# Function to create cosine S-bends.
def cosine_sbend(
    x0,
    y0,
    z0,
    wg_width,
    wg_thickness,
    bend_length,
    bend_height,
    medium,
    orientation="x",
    mirror=False,
    sidewall_angle=0,
):
    """Cosine S-bend function for smooth waveguide transitions."""
    cell = gdstk.Cell("bend")
    path = gdstk.RobustPath((x0, y0), wg_width, layer=1, datatype=0)

    if orientation == "x":
        path.segment(
            (x0 + bend_length, y0),
            offset=lambda u: -bend_height * np.cos(np.pi * u) / 2 + bend_height / 2,
        )
        if mirror:
            path.mirror((x0 + 1, y0), (x0, y0))
    elif orientation == "y":
        path.segment(
            (x0, y0 + bend_length),
            offset=lambda u: -bend_height * np.cos(np.pi * u) / 2 + bend_height / 2,
        )
        if mirror:
            path.mirror((x0, y0 + 1), (x0, y0))

    cell.add(path)
    bend_geo = td.PolySlab.from_gds(
        cell,
        gds_layer=1,
        axis=2,
        slab_bounds=(z0 - wg_thickness / 2, z0 + wg_thickness / 2),
        sidewall_angle=sidewall_angle,
    )
    return td.Structure(geometry=bend_geo[0], medium=medium)



def build_mmi_simulation(
    w_wg,
    h_si,
    w_mmi,
    l_mmi,
    gap,
    l_input,
    l_output,
    s_bend_offset,
    s_bend_length,
    mat_si,
    mat_sio2,
    lambda_0,
    lambda_min,
    lambda_max,
    lambda_step,
    plot=False,
):
    """Builds a complete 2x2 MMI power splitter simulation with cosine S-bends."""

    # ---- Simulation domain ----
    total_length = l_mmi + 2*s_bend_length + l_input + l_output + 2.0  # +2 µm margin
    x_margin = 5.0
    total_width = w_mmi + 2 * (s_bend_offset + 2.0)  # add a few µm margin
    total_height = 2.0
    sim_size_optimized = (total_length + x_margin, total_width, total_height)

    # ---- 1. MMI core ----
    mmi_section = td.Structure(
        geometry=td.Box(center=(0, 0, 0), size=(l_mmi, w_mmi, h_si)),
        medium=mat_si,
    )

    # ---- 2–5. Output branches ----
    y_offset = (w_mmi / 2) - (w_wg / 2)

    # Upper branch
    output_s_bend_1 = cosine_sbend(
        x0=l_mmi / 2,
        y0=+y_offset,
        z0=0,
        wg_width=w_wg,
        wg_thickness=h_si,
        bend_length=s_bend_length,
        bend_height=+s_bend_offset,
        medium=mat_si,
        orientation="x",
    )

    output_wg_1 = td.Structure(
        geometry=td.Box(
            center=(
                l_mmi / 2 + s_bend_length + l_output / 2,
                +y_offset + s_bend_offset,
                0,
            ),
            size=(l_output, w_wg, h_si),
        ),
        medium=mat_si,
    )

    # Lower branch
    output_s_bend_2 = cosine_sbend(
        x0=l_mmi / 2,
        y0=-y_offset,
        z0=0,
        wg_width=w_wg,
        wg_thickness=h_si,
        bend_length=s_bend_length,
        bend_height=-s_bend_offset,
        medium=mat_si,
        orientation="x",
    )

    output_wg_2 = td.Structure(
        geometry=td.Box(
            center=(
                l_mmi / 2 + s_bend_length + l_output / 2,
                -y_offset - s_bend_offset,
                0,
            ),
            size=(l_output, w_wg, h_si),
        ),
        medium=mat_si,
    )

    # ---- 6. Input branches (reflected) ----
    input_wg_1_geometry = output_wg_1.geometry.reflected(normal=(1, 0, 0))
    input_wg_2_geometry = output_wg_2.geometry.reflected(normal=(1, 0, 0))
    input_s_bend_1_geometry = output_s_bend_1.geometry.reflected(normal=(1, 0, 0))
    input_s_bend_2_geometry = output_s_bend_2.geometry.reflected(normal=(1, 0, 0))

    input_wg_1 = td.Structure(geometry=input_wg_1_geometry, medium=mat_si)
    input_wg_2 = td.Structure(geometry=input_wg_2_geometry, medium=mat_si)
    input_s_bend_1 = td.Structure(geometry=input_s_bend_1_geometry, medium=mat_si)
    input_s_bend_2 = td.Structure(geometry=input_s_bend_2_geometry, medium=mat_si)

    # ---- 7. Combine all ----
    mmi_structures = [
        input_s_bend_1,
        input_wg_1,
        input_s_bend_2,
        input_wg_2,
        mmi_section,
        output_s_bend_1,
        output_wg_1,
        output_s_bend_2,
        output_wg_2,
    ]

    ### --- base frequency --- ###

    freq0 = td.C_0 / lambda_0 
    # ---- 8. Frequency sweep ----
    wavelengths = np.arange(lambda_min, lambda_max + 0.5*lambda_step, lambda_step)
    frequencies = td.C_0 / wavelengths

    # ---- 9. Mode source ----
    source_position = (-(l_mmi/2 + s_bend_length + 0.5), +y_offset + s_bend_offset, 0)
    source_size = (0, 6 * w_wg, 6 * h_si)
    mode_source = td.ModeSource(
        center=source_position,
        size=source_size,
        source_time = td.GaussianPulse(freq0=freq0, fwidth=freq0/20),
        direction="+",
        mode_spec=td.ModeSpec(num_modes=1),
        mode_index=0,
    )

    monitor_1_position = (
        l_mmi / 2 + s_bend_length + 0.5,
        +y_offset + s_bend_offset,
        0,
    )

    monitor_2_position = (
        l_mmi / 2 + s_bend_length + 0.5,
        -y_offset - s_bend_offset,
        0,
    )

    monitor_size = (0, 6 * w_wg, 6 * h_si)

    mode_monitor_1 = td.ModeMonitor(
        center=monitor_1_position,
        size=monitor_size,
        freqs=frequencies,
        mode_spec=td.ModeSpec(num_modes=1),
        name="mode_output_1",
    )

    mode_monitor_2 = td.ModeMonitor(
        center=monitor_2_position,
        size=monitor_size,
        freqs=frequencies,
        mode_spec=td.ModeSpec(num_modes=1),
        name="mode_output_2",
    )

    # ---- 11. Field monitor ----
    field_freqs = [td.C_0 / 1.55, td.C_0 / 1.58]
    field_monitor = td.FieldMonitor(
        center=(0, 0, 0),
        size=(td.inf, td.inf, 0),
        freqs=field_freqs,
        name="field_xy",
    )


    def estimate_run_time_seconds(L_eff_um, lambda0_um, n_g=4.2, n_stop_periods=20):
        c_um_per_fs = 0.299792458
        t_flight_fs = n_g * L_eff_um / c_um_per_fs
        T0_fs = lambda0_um / c_um_per_fs
        return (t_flight_fs + n_stop_periods * T0_fs) * 1e-15
    

    # Effective optical path from source plane to monitor plane (two bends + MMI + straights)
    L_path_um = l_input + s_bend_length + l_mmi + s_bend_length + l_output
    # add a safety margin to account for PML spacing and weak cavities
    L_eff_um  = L_path_um + 6.0

    rt_est = estimate_run_time_seconds(L_eff_um, lambda_0, n_g=4.2, n_stop_periods=24)
    run_time_sec = max(rt_est, 5e-12)   # enforce ~5 ps minimum


    # ---- 12. Build simulation ----
    sim_mmi = td.Simulation(
        size=sim_size_optimized,
        structures=mmi_structures,
        sources=[mode_source],
        monitors=[mode_monitor_1, mode_monitor_2, field_monitor],
        run_time=run_time_sec,
        shutoff=1e-5,
        boundary_spec=td.BoundarySpec.all_sides(boundary=td.PML(num_layers=12)),
        medium=mat_sio2,
        grid_spec=td.GridSpec(
            grid_x=td.AutoGrid(min_steps_per_wvl=20),
            grid_y=td.AutoGrid(min_steps_per_wvl=20),
            grid_z=td.AutoGrid(min_steps_per_wvl=20),
            wavelength=lambda_0,
        ),
        symmetry=(0, 0, 0),
    )


    return (sim_mmi)
