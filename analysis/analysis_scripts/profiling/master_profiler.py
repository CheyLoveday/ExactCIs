#!/usr/bin/env python
"""
Master profiling coordinator for ExactCIs.

This script coordinates the complete profiling strategy by running:
1. Comprehensive function and line-level profiling
2. Memory usage analysis
3. Scalability and complexity analysis  
4. Generates unified reports with optimization recommendations

Usage:
    python master_profiler.py --full  # Complete analysis
    python master_profiler.py --quick # Quick analysis (subset of tests)
    python master_profiler.py --methods conditional midp blaker  # Specific methods only
"""

import sys
import os
import time
import argparse
import json
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import traceback

# Import individual profiling modules
from comprehensive_profiler import ComprehensiveProfiler
from memory_profiler import MemoryProfiler  
from scalability_analyzer import ScalabilityAnalyzer

class MasterProfiler:
    """Master coordinator for all ExactCIs profiling analyses."""
    
    def __init__(self, output_dir: str = "profiling_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Initialize individual profilers
        self.comprehensive_profiler = ComprehensiveProfiler(str(self.output_dir))
        self.memory_profiler = MemoryProfiler(str(self.output_dir))
        self.scalability_analyzer = ScalabilityAnalyzer(str(self.output_dir))
        
        # Results storage
        self.results = {
            "timestamp": self.timestamp,
            "comprehensive_analysis": {},
            "memory_analysis": {},
            "scalability_analysis": {},
            "unified_summary": {},
            "optimization_roadmap": {}
        }

    def run_quick_analysis(self, methods: Optional[List[str]] = None) -> Dict[str, Any]:
        """Run a quick profiling analysis with reduced scope."""
        print("=== ExactCIs Quick Profiling Analysis ===")
        print("Running abbreviated analysis for rapid feedback...\n")
        
        if methods is None:
            methods = ["conditional", "midp", "blaker"]  # Skip unconditional for quick analysis
        
        start_time = time.time()
        
        try:
            # Quick comprehensive profiling (function-level only)
            print("1. Running quick comprehensive profiling...")
            comprehensive_results = self.comprehensive_profiler.run_comprehensive_analysis(
                methods_to_profile=methods,
                include_line_profiling=False,
                include_scalability=False
            )
            self.results["comprehensive_analysis"] = comprehensive_results
            
            # Quick memory analysis (small tables only)
            print("\n2. Running quick memory analysis...")
            memory_results = self.memory_profiler.run_complete_memory_analysis(
                max_table_size=100,
                include_grid_analysis=False,
                include_method_comparison=True
            )
            self.results["memory_analysis"] = memory_results
            
            # Skip scalability analysis for quick mode
            self.results["scalability_analysis"] = {"skipped": "Quick analysis mode"}
            
        except Exception as e:
            print(f"Error in quick analysis: {e}")
            traceback.print_exc()
            return {"error": str(e)}
        
        # Generate quick summary
        self.results["unified_summary"] = self._generate_quick_summary()
        
        elapsed_time = time.time() - start_time
        print(f"\nQuick analysis completed in {elapsed_time:.1f} seconds")
        
        # Save results
        self._save_results("quick")
        return self.results

    def run_full_analysis(self, methods: Optional[List[str]] = None) -> Dict[str, Any]:
        """Run complete profiling analysis with all components."""
        print("=== ExactCIs Complete Profiling Analysis ===")
        print("Running comprehensive multi-level analysis...\n")
        
        if methods is None:
            methods = ["conditional", "midp", "blaker", "unconditional"]
        
        start_time = time.time()
        
        try:
            # 1. Comprehensive profiling
            print("1. Running comprehensive profiling analysis...")
            comprehensive_results = self.comprehensive_profiler.run_comprehensive_analysis(
                methods_to_profile=methods,
                include_line_profiling=True,
                include_scalability=True
            )
            self.results["comprehensive_analysis"] = comprehensive_results
            
            # 2. Memory profiling
            print("\n2. Running memory profiling analysis...")
            memory_results = self.memory_profiler.run_complete_memory_analysis(
                max_table_size=200,
                include_grid_analysis=True,
                include_method_comparison=True
            )
            self.results["memory_analysis"] = memory_results
            
            # 3. Scalability analysis
            print("\n3. Running scalability analysis...")
            scalability_results = self.scalability_analyzer.run_complete_scalability_analysis(
                methods_to_analyze=methods,
                max_size=250,
                include_complexity=True,
                include_parameters=True,
                include_comparison=True
            )
            self.results["scalability_analysis"] = scalability_results
            
        except Exception as e:
            print(f"Error in full analysis: {e}")
            traceback.print_exc()
            return {"error": str(e)}
        
        # Generate unified analysis
        self.results["unified_summary"] = self._generate_unified_summary()
        self.results["optimization_roadmap"] = self._generate_optimization_roadmap()
        
        elapsed_time = time.time() - start_time
        print(f"\nComplete analysis finished in {elapsed_time:.1f} seconds")
        
        # Save results and generate reports
        self._save_results("full")
        self._generate_html_report()
        
        return self.results

    def run_targeted_analysis(self, 
                             target_methods: List[str],
                             focus_areas: List[str] = None) -> Dict[str, Any]:
        """Run targeted analysis focusing on specific methods and areas."""
        
        if focus_areas is None:
            focus_areas = ["performance", "memory", "scalability"]
        
        print(f"=== ExactCIs Targeted Profiling Analysis ===")
        print(f"Target methods: {target_methods}")
        print(f"Focus areas: {focus_areas}\n")
        
        start_time = time.time()
        
        try:
            # Performance analysis
            if "performance" in focus_areas:
                print("1. Running targeted performance analysis...")
                comprehensive_results = self.comprehensive_profiler.run_comprehensive_analysis(
                    methods_to_profile=target_methods,
                    include_line_profiling=True,
                    include_scalability=False
                )
                self.results["comprehensive_analysis"] = comprehensive_results
            
            # Memory analysis
            if "memory" in focus_areas:
                print("\n2. Running targeted memory analysis...")
                memory_results = self.memory_profiler.run_complete_memory_analysis(
                    max_table_size=150,
                    include_grid_analysis="unconditional" in target_methods,
                    include_method_comparison=True
                )
                self.results["memory_analysis"] = memory_results
            
            # Scalability analysis
            if "scalability" in focus_areas:
                print("\n3. Running targeted scalability analysis...")
                scalability_results = self.scalability_analyzer.run_complete_scalability_analysis(
                    methods_to_analyze=target_methods,
                    max_size=200,
                    include_complexity=True,
                    include_parameters="unconditional" in target_methods,
                    include_comparison=len(target_methods) > 1
                )
                self.results["scalability_analysis"] = scalability_results
                
        except Exception as e:
            print(f"Error in targeted analysis: {e}")
            traceback.print_exc()
            return {"error": str(e)}
        
        # Generate targeted summary
        self.results["unified_summary"] = self._generate_targeted_summary(target_methods, focus_areas)
        
        elapsed_time = time.time() - start_time
        print(f"\nTargeted analysis completed in {elapsed_time:.1f} seconds")
        
        # Save results
        self._save_results("targeted")
        return self.results

    def _generate_quick_summary(self) -> Dict[str, Any]:
        """Generate summary for quick analysis."""
        summary = {
            "analysis_type": "quick",
            "key_findings": [],
            "immediate_recommendations": [],
            "performance_overview": {}
        }
        
        # Extract key performance metrics
        if "comprehensive_analysis" in self.results:
            comp_summary = self.results["comprehensive_analysis"].get("summary", {})
            if "performance_ranking" in comp_summary:
                summary["performance_overview"]["speed_ranking"] = comp_summary["performance_ranking"]
        
        # Extract memory efficiency
        if "memory_analysis" in self.results:
            mem_summary = self.results["memory_analysis"].get("summary", {})
            if "memory_efficiency_ranking" in mem_summary:
                summary["performance_overview"]["memory_ranking"] = mem_summary["memory_efficiency_ranking"]
        
        # Generate quick recommendations
        summary["immediate_recommendations"] = [
            "Run full analysis for detailed optimization guidance",
            "Focus on the slowest methods identified in performance ranking",
            "Consider memory usage patterns for large table scenarios"
        ]
        
        # Key findings
        summary["key_findings"] = [
            f"Quick analysis completed for basic performance assessment",
            f"Performance ranking established for {len(summary['performance_overview'].get('speed_ranking', []))} methods",
            "Memory efficiency patterns identified for optimization planning"
        ]
        
        return summary

    def _generate_unified_summary(self) -> Dict[str, Any]:
        """Generate unified summary across all analyses."""
        summary = {
            "analysis_type": "comprehensive",
            "overall_performance_assessment": {},
            "critical_bottlenecks": [],
            "optimization_priorities": [],
            "method_recommendations": {},
            "implementation_insights": []
        }
        
        # Combine performance data from all analyses
        performance_data = {}
        
        # From comprehensive analysis
        if "comprehensive_analysis" in self.results:
            comp_data = self.results["comprehensive_analysis"]
            if "summary" in comp_data and "performance_ranking" in comp_data["summary"]:
                performance_data["speed_ranking"] = comp_data["summary"]["performance_ranking"]
        
        # From memory analysis  
        if "memory_analysis" in self.results:
            mem_data = self.results["memory_analysis"]
            if "summary" in mem_data and "memory_efficiency_ranking" in mem_data["summary"]:
                performance_data["memory_ranking"] = mem_data["summary"]["memory_efficiency_ranking"]
        
        # From scalability analysis
        if "scalability_analysis" in self.results:
            scal_data = self.results["scalability_analysis"]
            if "summary" in scal_data:
                scal_summary = scal_data["summary"]
                performance_data["complexity_classes"] = scal_summary.get("complexity_classes", {})
                performance_data["scalability_limits"] = scal_summary.get("scalability_limits", {})
        
        summary["overall_performance_assessment"] = performance_data
        
        # Identify critical bottlenecks
        bottlenecks = []
        
        # Check for slow methods
        if "speed_ranking" in performance_data:
            slowest_methods = performance_data["speed_ranking"][-2:]  # Two slowest
            for method, time in slowest_methods:
                if time > 1.0:  # More than 1 second average
                    bottlenecks.append(f"{method} method: {time:.3f}s average execution time")
        
        # Check for poor memory efficiency
        if "memory_ranking" in performance_data:
            memory_ranking = performance_data["memory_ranking"]
            if memory_ranking:
                worst_memory = memory_ranking[-1]  # Worst memory efficiency
                bottlenecks.append(f"{worst_memory[0]} method: poor memory efficiency ({worst_memory[1]:.2f} MB/s)")
        
        # Check for poor scalability
        if "scalability_limits" in performance_data:
            for method, limit in performance_data["scalability_limits"].items():
                if limit < 100:
                    bottlenecks.append(f"{method} method: poor scalability (limit ~{limit})")
        
        summary["critical_bottlenecks"] = bottlenecks
        
        # Generate optimization priorities
        priorities = []
        
        if "unconditional" in str(bottlenecks):
            priorities.append("HIGH: Optimize unconditional method grid search algorithm")
        
        if "blaker" in str(bottlenecks):
            priorities.append("MEDIUM: Optimize Blaker's p-value calculations")
        
        priorities.extend([
            "LOW: Implement caching for core probability functions",
            "LOW: Consider parallel processing for grid-based methods"
        ])
        
        summary["optimization_priorities"] = priorities
        
        # Method-specific recommendations
        recommendations = {}
        
        for method in ["conditional", "midp", "blaker", "unconditional"]:
            method_recs = []
            
            # Check if method appears in bottlenecks
            if any(method in bottleneck for bottleneck in bottlenecks):
                if method == "unconditional":
                    method_recs.extend([
                        "Reduce default grid size for large tables",
                        "Implement adaptive grid refinement",
                        "Add early stopping criteria for convergence"
                    ])
                elif method == "blaker":
                    method_recs.extend([
                        "Optimize acceptability function calculations",
                        "Consider vectorized p-value computation",
                        "Cache intermediate results"
                    ])
                else:
                    method_recs.append("Profile at line level to identify specific bottlenecks")
            else:
                method_recs.append("Performance appears adequate for most use cases")
            
            recommendations[method] = method_recs
        
        summary["method_recommendations"] = recommendations
        
        # Implementation insights
        insights = [
            "Log-space arithmetic provides numerical stability but may impact performance",
            "Grid-based methods show polynomial scaling with table size",
            "Memory usage is generally efficient except for very large tables",
            "Parameter tuning can significantly impact unconditional method performance"
        ]
        
        summary["implementation_insights"] = insights
        
        return summary

    def _generate_targeted_summary(self, target_methods: List[str], focus_areas: List[str]) -> Dict[str, Any]:
        """Generate summary for targeted analysis."""
        summary = {
            "analysis_type": "targeted",
            "target_methods": target_methods,
            "focus_areas": focus_areas,
            "targeted_findings": {},
            "specific_recommendations": []
        }
        
        # Extract findings for each focus area
        for area in focus_areas:
            if area == "performance" and "comprehensive_analysis" in self.results:
                comp_data = self.results["comprehensive_analysis"]
                summary["targeted_findings"]["performance"] = {
                    "method_ranking": comp_data.get("summary", {}).get("performance_ranking", []),
                    "bottlenecks": comp_data.get("summary", {}).get("bottlenecks_identified", [])
                }
            
            elif area == "memory" and "memory_analysis" in self.results:
                mem_data = self.results["memory_analysis"]
                summary["targeted_findings"]["memory"] = {
                    "efficiency_ranking": mem_data.get("summary", {}).get("memory_efficiency_ranking", []),
                    "scaling_characteristics": mem_data.get("summary", {}).get("scaling_characteristics", {})
                }
            
            elif area == "scalability" and "scalability_analysis" in self.results:
                scal_data = self.results["scalability_analysis"]
                summary["targeted_findings"]["scalability"] = {
                    "complexity_classes": scal_data.get("summary", {}).get("complexity_classes", {}),
                    "limits": scal_data.get("summary", {}).get("scalability_limits", {})
                }
        
        # Generate specific recommendations for target methods
        for method in target_methods:
            if method == "unconditional":
                summary["specific_recommendations"].append(
                    f"{method}: Consider implementing adaptive grid sizing and timeout optimization"
                )
            elif method == "blaker":
                summary["specific_recommendations"].append(
                    f"{method}: Focus on optimizing p-value calculation loops"
                )
            else:
                summary["specific_recommendations"].append(
                    f"{method}: Performance appears adequate, consider caching optimizations"
                )
        
        return summary

    def _generate_optimization_roadmap(self) -> Dict[str, Any]:
        """Generate comprehensive optimization roadmap."""
        roadmap = {
            "immediate_actions": [],
            "short_term_optimizations": [],
            "long_term_improvements": [],
            "implementation_strategy": {},
            "expected_impact": {}
        }
        
        # Immediate actions (quick wins)
        roadmap["immediate_actions"] = [
            "Increase LRU cache sizes for core functions (pmf_weights, support)",
            "Reduce default grid_size for unconditional method on large tables",
            "Implement timeout warnings for long-running calculations",
            "Add progress monitoring for grid-based methods"
        ]
        
        # Short-term optimizations (1-2 weeks)
        roadmap["short_term_optimizations"] = [
            "Vectorize Blaker's acceptability function calculations",
            "Implement adaptive grid refinement for unconditional method",
            "Add memory-efficient streaming for large table calculations",
            "Optimize root-finding algorithms with better initial guesses"
        ]
        
        # Long-term improvements (months)
        roadmap["long_term_improvements"] = [
            "Implement parallel processing for independent grid computations",
            "Research and implement advanced numerical optimization techniques",
            "Consider C/Cython extensions for critical computational kernels",
            "Develop machine learning models for optimal parameter selection"
        ]
        
        # Implementation strategy
        roadmap["implementation_strategy"] = {
            "phase_1": {
                "duration": "1 week",
                "focus": "Cache optimization and parameter tuning",
                "methods": ["all"],
                "expected_speedup": "10-30%"
            },
            "phase_2": {
                "duration": "2-3 weeks", 
                "focus": "Algorithm optimization and vectorization",
                "methods": ["blaker", "unconditional"],
                "expected_speedup": "50-100%"
            },
            "phase_3": {
                "duration": "2-3 months",
                "focus": "Advanced optimization and parallelization",
                "methods": ["unconditional"],
                "expected_speedup": "200-500%"
            }
        }
        
        # Expected impact assessment
        roadmap["expected_impact"] = {
            "performance": "Significant improvements for large tables and grid-based methods",
            "memory": "Moderate improvements through streaming and efficient allocation",
            "scalability": "Major improvements enabling larger table computations",
            "user_experience": "Faster computations, better progress feedback, more reliable timeouts"
        }
        
        return roadmap

    def _save_results(self, analysis_type: str):
        """Save analysis results to JSON file."""
        filename = f"master_profiling_{analysis_type}_{self.timestamp}.json"
        filepath = self.output_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"\nResults saved to: {filepath}")

    def _generate_html_report(self):
        """Generate HTML report for easy viewing."""
        html_content = self._create_html_report_content()
        
        report_file = self.output_dir / f"profiling_report_{self.timestamp}.html"
        with open(report_file, 'w') as f:
            f.write(html_content)
        
        print(f"HTML report generated: {report_file}")

    def _create_html_report_content(self) -> str:
        """Create HTML content for the profiling report."""
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>ExactCIs Profiling Report - {self.timestamp}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
        .header {{ background: #f4f4f4; padding: 20px; border-radius: 5px; }}
        .section {{ margin: 30px 0; }}
        .subsection {{ margin: 20px 0 20px 20px; }}
        .highlight {{ background: #fff3cd; padding: 10px; border-radius: 3px; }}
        .critical {{ background: #f8d7da; padding: 10px; border-radius: 3px; }}
        .success {{ background: #d1edff; padding: 10px; border-radius: 3px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        code {{ background: #f4f4f4; padding: 2px 4px; border-radius: 3px; }}
        pre {{ background: #f8f8f8; padding: 15px; border-radius: 5px; overflow-x: auto; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>ExactCIs Performance Profiling Report</h1>
        <p><strong>Generated:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        <p><strong>Analysis Type:</strong> {self.results.get('unified_summary', {}).get('analysis_type', 'comprehensive')}</p>
    </div>
    
    <div class="section">
        <h2>Executive Summary</h2>
        {self._generate_html_executive_summary()}
    </div>
    
    <div class="section">
        <h2>Performance Assessment</h2>
        {self._generate_html_performance_section()}
    </div>
    
    <div class="section">
        <h2>Critical Bottlenecks</h2>
        {self._generate_html_bottlenecks_section()}
    </div>
    
    <div class="section">
        <h2>Optimization Roadmap</h2>
        {self._generate_html_roadmap_section()}
    </div>
    
    <div class="section">
        <h2>Detailed Analysis</h2>
        {self._generate_html_detailed_section()}
    </div>
</body>
</html>
"""
        return html

    def _generate_html_executive_summary(self) -> str:
        """Generate executive summary section for HTML report."""
        summary = self.results.get("unified_summary", {})
        
        key_findings = summary.get("key_findings", ["No key findings available"])
        critical_bottlenecks = summary.get("critical_bottlenecks", [])
        
        html = "<div class='subsection'>"
        
        if critical_bottlenecks:
            html += "<div class='critical'><h3>Critical Issues Found</h3><ul>"
            for bottleneck in critical_bottlenecks[:3]:  # Top 3
                html += f"<li>{bottleneck}</li>"
            html += "</ul></div>"
        else:
            html += "<div class='success'><h3>No Critical Performance Issues Detected</h3></div>"
        
        html += "<h3>Key Findings</h3><ul>"
        for finding in key_findings[:5]:  # Top 5
            html += f"<li>{finding}</li>"
        html += "</ul></div>"
        
        return html

    def _generate_html_performance_section(self) -> str:
        """Generate performance assessment section."""
        performance = self.results.get("unified_summary", {}).get("overall_performance_assessment", {})
        
        html = "<div class='subsection'>"
        
        # Speed ranking
        if "speed_ranking" in performance:
            html += "<h3>Method Speed Ranking (Fastest to Slowest)</h3><table><tr><th>Method</th><th>Average Time (seconds)</th></tr>"
            for method, time in performance["speed_ranking"]:
                html += f"<tr><td>{method}</td><td>{time:.4f}</td></tr>"
            html += "</table>"
        
        # Memory ranking
        if "memory_ranking" in performance:
            html += "<h3>Memory Efficiency Ranking</h3><table><tr><th>Method</th><th>Efficiency Score (MB/s)</th></tr>"
            for method, score in performance["memory_ranking"]:
                html += f"<tr><td>{method}</td><td>{score:.2f}</td></tr>"
            html += "</table>"
        
        # Scalability limits
        if "scalability_limits" in performance:
            html += "<h3>Scalability Limits</h3><table><tr><th>Method</th><th>Max Feasible Table Size</th></tr>"
            for method, limit in performance["scalability_limits"].items():
                html += f"<tr><td>{method}</td><td>~{limit}</td></tr>"
            html += "</table>"
        
        html += "</div>"
        return html

    def _generate_html_bottlenecks_section(self) -> str:
        """Generate bottlenecks section."""
        bottlenecks = self.results.get("unified_summary", {}).get("critical_bottlenecks", [])
        
        html = "<div class='subsection'>"
        
        if bottlenecks:
            html += "<div class='highlight'><h3>Identified Performance Bottlenecks</h3><ol>"
            for bottleneck in bottlenecks:
                html += f"<li>{bottleneck}</li>"
            html += "</ol></div>"
        else:
            html += "<div class='success'><p>No critical performance bottlenecks identified.</p></div>"
        
        html += "</div>"
        return html

    def _generate_html_roadmap_section(self) -> str:
        """Generate optimization roadmap section."""
        roadmap = self.results.get("optimization_roadmap", {})
        
        html = "<div class='subsection'>"
        
        # Immediate actions
        immediate = roadmap.get("immediate_actions", [])
        if immediate:
            html += "<h3>Immediate Actions (Quick Wins)</h3><ul>"
            for action in immediate:
                html += f"<li>{action}</li>"
            html += "</ul>"
        
        # Short-term optimizations
        short_term = roadmap.get("short_term_optimizations", [])
        if short_term:
            html += "<h3>Short-term Optimizations (1-2 weeks)</h3><ul>"
            for optimization in short_term:
                html += f"<li>{optimization}</li>"
            html += "</ul>"
        
        # Implementation strategy
        strategy = roadmap.get("implementation_strategy", {})
        if strategy:
            html += "<h3>Implementation Strategy</h3><table><tr><th>Phase</th><th>Duration</th><th>Focus</th><th>Expected Speedup</th></tr>"
            for phase, details in strategy.items():
                html += f"<tr><td>{phase}</td><td>{details.get('duration', 'TBD')}</td><td>{details.get('focus', 'TBD')}</td><td>{details.get('expected_speedup', 'TBD')}</td></tr>"
            html += "</table>"
        
        html += "</div>"
        return html

    def _generate_html_detailed_section(self) -> str:
        """Generate detailed analysis section."""
        html = "<div class='subsection'>"
        html += "<p>For detailed technical analysis, please refer to the individual profiling output files:</p><ul>"
        html += f"<li><code>comprehensive_analysis_{self.timestamp}.json</code> - Function and line-level profiling</li>"
        html += f"<li><code>memory_analysis_{self.timestamp}.json</code> - Memory usage patterns</li>"
        html += f"<li><code>scalability_analysis_{self.timestamp}.json</code> - Computational complexity analysis</li>"
        html += f"<li><code>master_profiling_full_{self.timestamp}.json</code> - Complete unified results</li>"
        html += "</ul></div>"
        return html


def main():
    parser = argparse.ArgumentParser(description="Master profiler for ExactCIs")
    parser.add_argument("--full", action="store_true", 
                       help="Run complete comprehensive analysis")
    parser.add_argument("--quick", action="store_true",
                       help="Run quick analysis for rapid feedback") 
    parser.add_argument("--targeted", action="store_true",
                       help="Run targeted analysis on specific methods")
    parser.add_argument("--methods", nargs="*",
                       choices=["conditional", "midp", "blaker", "unconditional", "wald"],
                       help="Specific methods to analyze")
    parser.add_argument("--focus", nargs="*", 
                       choices=["performance", "memory", "scalability"],
                       help="Focus areas for targeted analysis")
    parser.add_argument("--output-dir", default="profiling_results",
                       help="Output directory for results")
    
    args = parser.parse_args()
    
    # Create master profiler
    profiler = MasterProfiler(args.output_dir)
    
    # Determine analysis type
    if args.quick:
        analysis_type = "quick"
    elif args.targeted:
        analysis_type = "targeted"
    elif args.full:
        analysis_type = "full"
    else:
        # Default to quick if no specific type requested
        analysis_type = "quick"
        print("No analysis type specified, defaulting to --quick")
    
    print(f"Starting {analysis_type} profiling analysis...")
    
    try:
        if analysis_type == "quick":
            results = profiler.run_quick_analysis(args.methods)
        elif analysis_type == "targeted":
            if not args.methods:
                print("Error: --targeted requires --methods to be specified")
                return 1
            results = profiler.run_targeted_analysis(args.methods, args.focus)
        elif analysis_type == "full":
            results = profiler.run_full_analysis(args.methods)
        
        if "error" in results:
            print(f"Analysis failed: {results['error']}")
            return 1
        
        print("\n" + "="*60)
        print(f"{analysis_type.upper()} PROFILING ANALYSIS COMPLETED SUCCESSFULLY")
        print("="*60)
        print(f"Results available in: {profiler.output_dir}")
        
        return 0
        
    except Exception as e:
        print(f"\nUnexpected error in master profiler: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())