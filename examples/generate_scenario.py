"""
Generate Building Scenarios
============================

This example demonstrates how to generate building scenarios
for energy analysis.
"""

# Assuming that cluster_1_limits is defined as:
cluster_1_limits = {
    'average_opaque_surface_transmittance': {'min': 0.15, 'max': 2.8},  # Example values
    'average_glazed_surface_transmittance': {'min': 1.2, 'max': 5.5}   # Example values
}

import numpy as np
import pandas as pd

# Extract limits for better readability
opaque_min = cluster_1_limits['average_opaque_surface_transmittance']['min']
opaque_max = cluster_1_limits['average_opaque_surface_transmittance']['max']
glazed_min = cluster_1_limits['average_glazed_surface_transmittance']['min']
glazed_max = cluster_1_limits['average_glazed_surface_transmittance']['max']

print(f"Range opaque transmittance: {opaque_min:.3f} - {opaque_max:.3f}")
print(f"Range glazed transmittance: {glazed_min:.3f} - {glazed_max:.3f}")

# Generate 10 strategic scenarios for energy efficiency
list_dict_scenarios = [
    # 1. OPTIMAL SCENARIO - Maximum efficiency (minimum values)
    {
        'name': 'High Efficiency Optimal', 
        'parameters': {
            'average_opaque_surface_transmittance': opaque_min,
            'average_glazed_surface_transmittance': glazed_min
        }
    },
    
    # 2. WORST SCENARIO - Minimum efficiency (maximum values)
    {
        'name': 'Low Efficiency Worst', 
        'parameters': {
            'average_opaque_surface_transmittance': opaque_max,
            'average_glazed_surface_transmittance': glazed_max
        }
    },
    
    # 3. BALANCED SCENARIO - Intermediate values
    {
        'name': 'Balanced Performance', 
        'parameters': {
            'average_opaque_surface_transmittance': (opaque_min + opaque_max) / 2,
            'average_glazed_surface_transmittance': (glazed_min + glazed_max) / 2
        }
    },
    
    # 4. ENVELOPE OPTIMIZED - Efficient envelope, standard glazed
    {
        'name': 'Optimized Envelope', 
        'parameters': {
            'average_opaque_surface_transmittance': opaque_min + (opaque_max - opaque_min) * 0.2,
            'average_glazed_surface_transmittance': glazed_min + (glazed_max - glazed_min) * 0.4
        }
    },
    
    # 5. GLAZED ADVANCED - Efficient glazed, standard envelope  
    {
        'name': 'Advanced Glazing', 
        'parameters': {
            'average_opaque_surface_transmittance': opaque_min + (opaque_max - opaque_min) * 0.4,
            'average_glazed_surface_transmittance': glazed_min + (glazed_max - glazed_min) * 0.2
        }
    },
    
    # 6. CONSERVATIVE SCENARIO - Moderate improvement
    {
        'name': 'Conservative Upgrade', 
        'parameters': {
            'average_opaque_surface_transmittance': opaque_min + (opaque_max - opaque_min) * 0.3,
            'average_glazed_surface_transmittance': glazed_min + (glazed_max - glazed_min) * 0.3
        }
    },
    
    # 7. AGGRESSIVE SCENARIO - High efficiency
    {
        'name': 'Aggressive Efficiency', 
        'parameters': {
            'average_opaque_surface_transmittance': opaque_min + (opaque_max - opaque_min) * 0.1,
            'average_glazed_surface_transmittance': glazed_min + (glazed_max - glazed_min) * 0.15
        }
    },
    
    # 8. CURRENT MARKET STANDARD - Typical market performance
    {
        'name': 'Current Market Standard', 
        'parameters': {
            'average_opaque_surface_transmittance': opaque_min + (opaque_max - opaque_min) * 0.6,
            'average_glazed_surface_transmittance': glazed_min + (glazed_max - glazed_min) * 0.5
        }
    },
    
    # 9. ECONOMIC RETROFIT SCENARIO - Cost-effective improvement
    {
        'name': 'Economic Retrofit', 
        'parameters': {
            'average_opaque_surface_transmittance': opaque_min + (opaque_max - opaque_min) * 0.45,
            'average_glazed_surface_transmittance': glazed_min + (glazed_max - glazed_min) * 0.35
        }
    },
    
    # 10. HIGH PERFORMANCE SCENARIO - Quasi ottimale ma realistico
    {
        'name': 'High Performance Realistic', 
        'parameters': {
            'average_opaque_surface_transmittance': opaque_min + (opaque_max - opaque_min) * 0.15,
            'average_glazed_surface_transmittance': glazed_min + (glazed_max - glazed_min) * 0.25
        }
    }
]
# Save in a file
list_dict_scenarios = pd.DataFrame(list_dict_scenarios)
list_dict_scenarios.to_csv("/Users/dantonucci/Documents/gitLab/pybuildingcluster/examples/example_results/list_dict_scenarios.csv", index=False)

# Print generated scenarios with details
print("\n" + "="*80)
print("SCENARIOS GENERATED FOR ENERGY EFFICIENCY ANALYSIS")
print("="*80)
print("📊 Values low = BETTER efficiency | Values high = WORSE efficiency")
print("="*80)

for i, scenario in enumerate(list_dict_scenarios, 1):
    opaque_val = scenario['parameters']['average_opaque_surface_transmittance']
    glazed_val = scenario['parameters']['average_glazed_surface_transmittance']
    
    # Calculate percentiles for interpretation
    opaque_percentile = (opaque_val - opaque_min) / (opaque_max - opaque_min) * 100
    glazed_percentile = (glazed_val - glazed_min) / (glazed_max - glazed_min) * 100
    
    # Determine efficiency level
    avg_percentile = (opaque_percentile + glazed_percentile) / 2
    if avg_percentile <= 20:
        efficiency_level = "🟢 ECCELLENTE"
    elif avg_percentile <= 40:
        efficiency_level = "🔵 BUONA"
    elif avg_percentile <= 60:
        efficiency_level = "🟡 MEDIA"
    elif avg_percentile <= 80:
        efficiency_level = "🟠 BASSA"
    else:
        efficiency_level = "🔴 SCARSA"
    
    print(f"\n{i:2d}. {scenario['name']}")
    print(f"    Opaque Transmittance: {opaque_val:.3f} ({opaque_percentile:.0f}° percentile)")
    print(f"    Glazed Transmittance: {glazed_val:.3f} ({glazed_percentile:.0f}° percentile)")
    print(f"    Energy Efficiency: {efficiency_level}")

# Verify the distribution of scenarios
print(f"\n" + "="*50)
print("📈 ANALISI DISTRIBUZIONE SCENARI")
print("="*50)

opaque_values = [s['parameters']['average_opaque_surface_transmittance'] for s in list_dict_scenarios]
glazed_values = [s['parameters']['average_glazed_surface_transmittance'] for s in list_dict_scenarios]

print(f"Opaque Transmittance:")
print(f"  Min scenario: {min(opaque_values):.3f}")
print(f"  Max scenario: {max(opaque_values):.3f}")
print(f"  Media scenari: {np.mean(opaque_values):.3f}")

print(f"\nGlazed Transmittance:")
print(f"  Min scenario: {min(glazed_values):.3f}")
print(f"  Max scenario: {max(glazed_values):.3f}")
print(f"  Media scenari: {np.mean(glazed_values):.3f}")

print(f"\n💡 SUGGERIMENTI:")
print(f"   • Usa 'High Efficiency Optimal' come benchmark migliore")
print(f"   • Confronta 'Current Market Standard' con target di miglioramento")
print(f"   • 'Economic Retrofit' rappresenta un upgrade realistico")
print(f"   • 'Low Efficiency Worst' identifica scenari da evitare")

# Output finale da usare nel codice
print(f"\n" + "="*50)
print("🔧 CODICE PER COPY-PASTE:")
print("="*50)
print("list_dict_scenarios = [")
for scenario in list_dict_scenarios:
    opaque_val = scenario['parameters']['average_opaque_surface_transmittance']
    glazed_val = scenario['parameters']['average_glazed_surface_transmittance']
    print(f"    {{'name': '{scenario['name']}', 'parameters': {{'average_opaque_surface_transmittance': {opaque_val:.3f}, 'average_glazed_surface_transmittance': {glazed_val:.3f}}}}},")
print("]")