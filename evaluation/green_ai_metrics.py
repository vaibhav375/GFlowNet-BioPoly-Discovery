"""
Green AI Metrics Module - Comprehensive Environmental Impact Assessment
========================================================================
Tracks carbon, energy, water, compute efficiency, and environmental benefit.
"""

import os
import json
import time
import logging
import platform
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional
from datetime import datetime

import numpy as np

logger = logging.getLogger(__name__)

# --- Constants ---

WATER_INTENSITY = {
    'us_average': 1.8, 'eu_average': 1.2, 'india': 2.5,
    'nordic': 0.5, 'renewable': 0.3,
}

EMBODIED_CARBON = {
    'cpu_laptop': 150, 'mps': 200, 'rtx_3060': 60, 'rtx_3080': 85,
    'rtx_3090': 100, 'rtx_4090': 120, 'a100': 150, 'v100': 120, 't4': 50,
}

HARDWARE_LIFETIME_HOURS = {
    'cpu_laptop': 20000, 'mps': 25000, 'rtx_3060': 30000, 'rtx_3080': 30000,
    'rtx_3090': 30000, 'rtx_4090': 30000, 'a100': 40000, 'v100': 40000, 't4': 40000,
}

PLASTIC_STATS = {
    'global_production_mt_per_year': 400,
    'ocean_leakage_mt_per_year': 11,
    'avg_degradation_years': 450,
    'co2_per_tonne_plastic': 6.0,
    'marine_species_affected': 700,
}


# --- GreenAIReport ---

@dataclass
class GreenAIReport:
    """Comprehensive Green AI metrics report."""
    project_name: str = "GFlowNet_BioPoly_Discovery"
    total_emissions_kg_co2: float = 0.0
    energy_consumed_kwh: float = 0.0
    training_time_hours: float = 0.0
    water_consumed_liters: float = 0.0
    water_intensity_region: str = "us_average"
    embodied_carbon_share_kg: float = 0.0
    total_lifecycle_carbon_kg: float = 0.0
    total_flops: float = 0.0
    flops_per_molecule: float = 0.0
    gpu_hours_per_candidate: float = 0.0
    molecules_per_kwh: float = 0.0
    total_parameters: int = 0
    parameters_per_valid_molecule: float = 0.0
    model_size_mb: float = 0.0
    total_molecules_generated: int = 0
    valid_molecules_generated: int = 0
    unique_molecules_generated: int = 0
    emissions_per_molecule: float = 0.0
    carbon_efficiency: float = 0.0
    total_epochs: int = 0
    total_steps: int = 0
    hardware: str = "CPU"
    gpu_model: str = "N/A"
    region: str = "Unknown"
    renewable_energy_pct: float = 0.0
    mixed_precision: bool = False
    gradient_checkpointing: bool = False
    early_stopping_used: bool = False
    carbon_budget_used: bool = False
    efficient_architecture: bool = True
    transfer_learning: bool = False
    data_efficient: bool = True
    time_saved_pct: float = 0.0
    energy_saved_pct: float = 0.0
    potential_plastic_replaced_tonnes: float = 0.0
    potential_co2_avoided_tonnes: float = 0.0
    potential_ocean_plastic_reduction_kg: float = 0.0
    biodeg_improvement_factor: float = 0.0
    baseline_emissions: Dict = field(default_factory=dict)
    baseline_comparison: Dict = field(default_factory=dict)
    sustainability_score: float = 0.0

    def compute_derived_metrics(self):
        """Compute all derived metrics from raw values."""
        if self.total_molecules_generated > 0:
            self.emissions_per_molecule = self.total_emissions_kg_co2 / self.total_molecules_generated
            if self.total_flops > 0:
                self.flops_per_molecule = self.total_flops / self.total_molecules_generated
        if self.total_emissions_kg_co2 > 0:
            self.carbon_efficiency = self.valid_molecules_generated / self.total_emissions_kg_co2
        if self.energy_consumed_kwh > 0:
            self.molecules_per_kwh = self.valid_molecules_generated / self.energy_consumed_kwh
        if self.training_time_hours > 0 and self.total_molecules_generated > 0:
            self.gpu_hours_per_candidate = self.training_time_hours / self.total_molecules_generated
        if self.valid_molecules_generated > 0 and self.total_parameters > 0:
            self.parameters_per_valid_molecule = self.total_parameters / self.valid_molecules_generated
        # Water footprint
        wi = WATER_INTENSITY.get(self.water_intensity_region, 1.8)
        self.water_consumed_liters = self.energy_consumed_kwh * wi
        # Lifecycle carbon
        self.total_lifecycle_carbon_kg = self.total_emissions_kg_co2 + self.embodied_carbon_share_kg
        # Environmental benefit potential
        if self.biodeg_improvement_factor > 1:
            frac = 0.00001
            self.potential_plastic_replaced_tonnes = (
                PLASTIC_STATS['global_production_mt_per_year'] * 1e6 * frac
            )
            self.potential_co2_avoided_tonnes = (
                self.potential_plastic_replaced_tonnes * PLASTIC_STATS['co2_per_tonne_plastic']
            )
            self.potential_ocean_plastic_reduction_kg = (
                PLASTIC_STATS['ocean_leakage_mt_per_year'] * 1e9 * frac
            )
        self._compute_sustainability_score()
        self._compute_baseline_comparison()

    def _compute_sustainability_score(self):
        """0-100 sustainability score."""
        score = 0.0
        # Carbon efficiency component (25 pts)
        if self.carbon_efficiency > 0:
            score += min(np.log10(max(self.carbon_efficiency, 1)) / 4 * 25, 25)
        # Green practices component (25 pts)
        practices = [
            self.mixed_precision, self.gradient_checkpointing,
            self.early_stopping_used, self.carbon_budget_used,
            self.efficient_architecture, self.transfer_learning,
            self.data_efficient,
        ]
        score += sum(practices) * (25 / len(practices))
        # Environmental impact component (25 pts)
        if self.biodeg_improvement_factor > 1:
            score += min(self.biodeg_improvement_factor / 100 * 25, 25)
        # Energy efficiency component (25 pts)
        if self.molecules_per_kwh > 0:
            score += min(np.log10(max(self.molecules_per_kwh, 1)) / 3 * 25, 25)
        self.sustainability_score = min(score, 100)

    def _compute_baseline_comparison(self):
        """Compare against baseline methods."""
        baselines = estimate_baseline_emissions()
        self.baseline_comparison = {}
        for method, bd in baselines.items():
            comp = {}
            be = bd['emissions_kg_co2']
            if be > 0:
                comp['carbon_reduction_pct'] = (1 - self.total_emissions_kg_co2 / be) * 100
            beff = bd.get('efficiency_molecules_per_kg', 0)
            if beff > 0 and self.carbon_efficiency > 0:
                comp['efficiency_ratio'] = self.carbon_efficiency / beff
            ben = bd.get('energy_kwh', 0)
            if ben > 0:
                comp['energy_reduction_pct'] = (1 - self.energy_consumed_kwh / ben) * 100
            self.baseline_comparison[method] = comp

    def to_dict(self) -> Dict:
        return asdict(self)

    def save(self, path: str):
        """Save report as JSON."""
        self.compute_derived_metrics()
        dirn = os.path.dirname(path)
        if dirn:
            os.makedirs(dirn, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        logger.info(f"Green AI report saved to {path}")

    def __str__(self):
        self.compute_derived_metrics()
        lines = [
            "", "=" * 70,
            "  COMPREHENSIVE GREEN AI METRICS REPORT",
            "=" * 70, "",
            f"  Project:   {self.project_name}",
            f"  Hardware:  {self.hardware} ({self.gpu_model})",
            f"  Region:    {self.region} ({self.renewable_energy_pct:.0f}% renewable)",
            f"  Date:      {datetime.now().strftime('%Y-%m-%d %H:%M')}", "",
            "  --- ENVIRONMENTAL IMPACT ---",
            f"  CO2 (operational):     {self.total_emissions_kg_co2:.4f} kg",
            f"  CO2 (embodied):        {self.embodied_carbon_share_kg:.4f} kg",
            f"  CO2 (lifecycle):       {self.total_lifecycle_carbon_kg:.4f} kg",
            f"  Energy consumed:       {self.energy_consumed_kwh:.4f} kWh",
            f"  Water footprint:       {self.water_consumed_liters:.2f} L",
            f"  Training time:         {self.training_time_hours:.2f} hours", "",
            "  --- COMPUTE EFFICIENCY ---",
            f"  Molecules generated:   {self.total_molecules_generated:,}",
            f"  Valid molecules:       {self.valid_molecules_generated:,}",
            f"  Unique molecules:      {self.unique_molecules_generated:,}",
            f"  Emissions/molecule:    {self.emissions_per_molecule:.6f} kg",
            f"  Molecules/kg CO2:     {self.carbon_efficiency:.0f}",
            f"  Molecules/kWh:        {self.molecules_per_kwh:.0f}", "",
            "  --- MODEL EFFICIENCY ---",
            f"  Parameters:            {self.total_parameters:,}",
            f"  Model size:            {self.model_size_mb:.2f} MB", "",
            "  --- GREEN AI PRACTICES ---",
            f"  Efficient architecture: {'YES' if self.efficient_architecture else 'NO'}",
            f"  Early stopping:         {'YES' if self.early_stopping_used else 'NO'}",
            f"  Data-efficient (AL):    {'YES' if self.data_efficient else 'NO'}",
            f"  Carbon budget:          {'YES' if self.carbon_budget_used else 'NO'}",
            f"  Mixed precision:        {'YES' if self.mixed_precision else 'NO'}",
            f"  Gradient checkpointing: {'YES' if self.gradient_checkpointing else 'NO'}",
            f"  Transfer learning:      {'YES' if self.transfer_learning else 'NO'}",
        ]
        if self.baseline_comparison:
            lines.append("")
            lines.append("  --- COMPARISON WITH BASELINES ---")
            for method, comp in self.baseline_comparison.items():
                cr = comp.get('carbon_reduction_pct', 0)
                er = comp.get('energy_reduction_pct', 0)
                lines.append(
                    f"  vs {method:<20} CO2:{cr:>+6.1f}%  Energy:{er:>+6.1f}%"
                )
        if self.potential_plastic_replaced_tonnes > 0:
            lines.append("")
            lines.append("  --- ENVIRONMENTAL BENEFIT POTENTIAL ---")
            lines.append(f"  Biodegradation speedup: {self.biodeg_improvement_factor:.0f}x")
            lines.append(f"  Plastic replaced:       {self.potential_plastic_replaced_tonnes:.1f} tonnes/yr")
            lines.append(f"  CO2 avoided:            {self.potential_co2_avoided_tonnes:.1f} tonnes/yr")
            lines.append(f"  Ocean plastic reduced:  {self.potential_ocean_plastic_reduction_kg:.0f} kg/yr")
        lines.append("")
        lines.append(f"  SUSTAINABILITY SCORE: {self.sustainability_score:.0f} / 100")
        lines.append("")
        lines.append("=" * 70)
        return "\n".join(lines)


# --- GreenAITracker ---

class GreenAITracker:
    """Comprehensive environmental impact tracker."""

    HARDWARE_TDP = {
        'cpu': 65, 'cpu_laptop': 25, 'mps': 30,
        'rtx_3060': 170, 'rtx_3080': 320, 'rtx_3090': 350,
        'rtx_4090': 450, 'a100': 300, 'v100': 300, 't4': 70,
    }

    CARBON_INTENSITY = {
        'us_average': 0.42, 'eu_average': 0.33, 'india': 0.82,
        'france': 0.05, 'canada': 0.12, 'china': 0.58, 'renewable': 0.02,
    }

    def __init__(self, hardware='cpu', region='us_average', pue=1.1):
        self.hardware = hardware
        self.region = region
        self.pue = pue
        self.tdp_watts = self.HARDWARE_TDP.get(hardware, 65)
        self.carbon_intensity = self.CARBON_INTENSITY.get(region, 0.42)
        self.start_time = None
        self.total_time_seconds = 0.0
        self.is_running = False
        self.phase_times = {}
        self._current_phase = None
        self._phase_start = None
        self.total_flops = 0.0
        self._codecarbon_tracker = None
        self._use_codecarbon = False
        self._init_codecarbon()

    def _init_codecarbon(self):
        """Try to initialise CodeCarbon; fall back to TDP estimate."""
        try:
            import subprocess
            skip = False
            if platform.system() == 'Darwin':
                try:
                    r = subprocess.run(
                        ['sudo', '-n', 'true'],
                        capture_output=True, timeout=2,
                    )
                    if r.returncode != 0:
                        skip = True
                except Exception:
                    skip = True
                if skip:
                    logger.info(
                        "macOS without passwordless sudo: "
                        "using TDP-based carbon estimation (accurate enough)"
                    )
            if not skip:
                from codecarbon import EmissionsTracker
                os.makedirs('./carbon_logs', exist_ok=True)
                self._codecarbon_tracker = EmissionsTracker(
                    project_name="GFlowNet_BioPoly_Discovery",
                    output_dir="./carbon_logs",
                    log_level="warning",
                    save_to_file=True,
                    allow_multiple_runs=True,
                    measure_power_secs=30,
                )
                self._use_codecarbon = True
        except Exception as e:
            logger.warning(f"CodeCarbon unavailable ({e}); using TDP estimation")

    # --- lifecycle ---

    def start(self):
        self.start_time = time.time()
        self.is_running = True
        if self._use_codecarbon and self._codecarbon_tracker:
            try:
                self._codecarbon_tracker.start()
            except Exception:
                self._use_codecarbon = False

    def stop(self):
        if self.is_running and self.start_time is not None:
            self.total_time_seconds += time.time() - self.start_time
            self.is_running = False
        if self._current_phase and self._phase_start:
            elapsed = time.time() - self._phase_start
            self.phase_times[self._current_phase] = (
                self.phase_times.get(self._current_phase, 0) + elapsed
            )
            self._current_phase = None
        if self._use_codecarbon and self._codecarbon_tracker:
            try:
                real = self._codecarbon_tracker.stop()
                if real and real > 0:
                    return real
            except Exception:
                pass
        return self.get_emissions()

    # --- phase tracking ---

    def start_phase(self, name: str):
        if self._current_phase and self._phase_start:
            elapsed = time.time() - self._phase_start
            self.phase_times[self._current_phase] = (
                self.phase_times.get(self._current_phase, 0) + elapsed
            )
        self._current_phase = name
        self._phase_start = time.time()

    def end_phase(self):
        if self._current_phase and self._phase_start:
            elapsed = time.time() - self._phase_start
            self.phase_times[self._current_phase] = (
                self.phase_times.get(self._current_phase, 0) + elapsed
            )
            self._current_phase = None
            self._phase_start = None

    # --- FLOPs ---

    def add_flops(self, flops: float):
        self.total_flops += flops

    def estimate_flops(self, model_params=0, batch_size=1, num_steps=1):
        """Estimate FLOPs: ~6 * params * batch * steps (forward+backward)."""
        flops = 6.0 * model_params * batch_size * num_steps
        self.total_flops += flops
        return flops

    # --- getters ---

    def get_emissions(self) -> float:
        """kg CO2 (TDP-based)."""
        hours = self.total_time_seconds / 3600.0
        energy_kwh = (self.tdp_watts / 1000.0) * hours * self.pue
        return energy_kwh * self.carbon_intensity

    def get_energy_kwh(self) -> float:
        if self._use_codecarbon and self._codecarbon_tracker:
            try:
                if (hasattr(self._codecarbon_tracker, '_total_energy')
                        and self._codecarbon_tracker._total_energy):
                    return self._codecarbon_tracker._total_energy.kWh
            except Exception:
                pass
        hours = self.total_time_seconds / 3600.0
        return (self.tdp_watts / 1000.0) * hours * self.pue

    def get_water_liters(self) -> float:
        return self.get_energy_kwh() * WATER_INTENSITY.get(self.region, 1.8)

    def get_embodied_carbon_share(self) -> float:
        total_kg = EMBODIED_CARBON.get(self.hardware, 100)
        lifetime_h = HARDWARE_LIFETIME_HOURS.get(self.hardware, 25000)
        return total_kg * (self.total_time_seconds / 3600.0 / lifetime_h)

    def get_phase_breakdown(self) -> Dict:
        total_tracked = sum(self.phase_times.values()) or 1.0
        breakdown = {}
        for phase, secs in self.phase_times.items():
            frac = secs / max(self.total_time_seconds, 1.0)
            breakdown[phase] = {
                'time_seconds': round(secs, 2),
                'time_pct': round((secs / total_tracked) * 100, 1),
                'energy_kwh': round(self.get_energy_kwh() * frac, 6),
                'emissions_kg': round(self.get_emissions() * frac, 6),
                'water_liters': round(self.get_water_liters() * frac, 4),
            }
        return breakdown

    def get_report(self) -> GreenAIReport:
        """Build a GreenAIReport from tracked data."""
        return GreenAIReport(
            total_emissions_kg_co2=self.get_emissions(),
            energy_consumed_kwh=self.get_energy_kwh(),
            training_time_hours=self.total_time_seconds / 3600.0,
            hardware=self.hardware,
            region=self.region,
            water_consumed_liters=self.get_water_liters(),
            water_intensity_region=self.region,
            embodied_carbon_share_kg=self.get_embodied_carbon_share(),
            total_flops=self.total_flops,
            baseline_emissions={
                'JT-VAE': 2.5, 'GA': 1.8, 'PPO': 1.5, 'Random': 0.05,
            },
        )


# --- Helper functions ---

def estimate_baseline_emissions() -> Dict:
    """Baseline method emissions for comparison in the paper."""
    return {
        'JT-VAE': {
            'emissions_kg_co2': 2.5,
            'energy_kwh': 8.3,
            'training_hours': 12.0,
            'efficiency_molecules_per_kg': 400,
            'hardware': 'V100',
            'source': 'Estimated: 12 h on V100',
        },
        'Genetic_Algorithm': {
            'emissions_kg_co2': 1.8,
            'energy_kwh': 6.1,
            'training_hours': 8.0,
            'efficiency_molecules_per_kg': 556,
            'hardware': 'CPU cluster',
            'source': 'Estimated: 8 h on CPU cluster',
        },
        'PPO_RL': {
            'emissions_kg_co2': 1.5,
            'energy_kwh': 5.0,
            'training_hours': 6.0,
            'efficiency_molecules_per_kg': 667,
            'hardware': 'V100',
            'source': 'Estimated: 6 h on V100',
        },
        'Random_Search': {
            'emissions_kg_co2': 0.05,
            'energy_kwh': 0.15,
            'training_hours': 0.5,
            'efficiency_molecules_per_kg': 200,
            'hardware': 'CPU',
            'source': 'Random SMILES enumeration',
        },
    }


def compute_environmental_benefit(
    avg_biodeg_months: float,
    num_candidates: int,
    target_degradation_years: float = 450,
) -> Dict:
    """Compute potential environmental benefit of discovered polymers."""
    target_months = target_degradation_years * 12
    improvement = target_months / max(avg_biodeg_months, 1)
    frac = 0.00001 * min(num_candidates / 10, 1.0)
    prod = PLASTIC_STATS['global_production_mt_per_year'] * 1e6  # tonnes
    replaced = prod * frac
    co2 = replaced * PLASTIC_STATS['co2_per_tonne_plastic']
    ocean = PLASTIC_STATS['ocean_leakage_mt_per_year'] * 1e6 * frac  # tonnes
    return {
        'biodeg_improvement_factor': improvement,
        'replacement_fraction': frac,
        'plastic_replaced_tonnes_per_year': replaced,
        'co2_avoided_tonnes_per_year': co2,
        'ocean_plastic_reduction_tonnes_per_year': ocean,
        'marine_species_benefited': int(
            PLASTIC_STATS['marine_species_affected'] * min(frac * 100, 1)
        ),
    }


def generate_green_ai_summary_table(report: GreenAIReport) -> str:
    """Generate Markdown summary table for the IEEE paper."""
    report.compute_derived_metrics()
    lines = [
        "| Metric | Value | Unit |",
        "|--------|-------|------|",
        f"| Operational CO2 | {report.total_emissions_kg_co2:.4f} | kg |",
        f"| Embodied CO2 | {report.embodied_carbon_share_kg:.4f} | kg |",
        f"| Lifecycle CO2 | {report.total_lifecycle_carbon_kg:.4f} | kg |",
        f"| Energy consumed | {report.energy_consumed_kwh:.4f} | kWh |",
        f"| Water footprint | {report.water_consumed_liters:.2f} | L |",
        f"| Training time | {report.training_time_hours:.2f} | hours |",
        f"| Total FLOPs | {report.total_flops:.2e} | - |",
        f"| Valid molecules | {report.valid_molecules_generated:,} | - |",
        f"| Carbon efficiency | {report.carbon_efficiency:.0f} | mol/kg CO2 |",
        f"| Molecules/kWh | {report.molecules_per_kwh:.0f} | mol/kWh |",
        f"| Sustainability score | {report.sustainability_score:.0f} | /100 |",
    ]
    if report.baseline_comparison:
        lines.append("")
        lines.append("**Comparison with baselines:**")
        lines.append("")
        lines.append("| Method | CO2 reduction | Energy reduction |")
        lines.append("|--------|--------------|-----------------|")
        for method, comp in report.baseline_comparison.items():
            cr = comp.get('carbon_reduction_pct', 0)
            er = comp.get('energy_reduction_pct', 0)
            lines.append(f"| {method} | {cr:+.1f}% | {er:+.1f}% |")
    return "\n".join(lines)
