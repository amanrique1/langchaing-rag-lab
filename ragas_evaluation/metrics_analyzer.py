"""
Advanced metrics analysis and reporting
"""
import pandas as pd
from typing import Dict, Any, List, Optional
from pathlib import Path
import json

from .config import EvaluationConfig

class MetricsAnalyzer:
    """Analyze and visualize RAGAS evaluation metrics"""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.thresholds = config.thresholds
        
    def analyze_results(self, results: Any) -> Dict[str, Any]:
        """
        Comprehensive analysis of evaluation results
        
        Args:
            results: RAGAS evaluation results
            
        Returns:
            Dictionary with detailed analysis
        """
        # Convert to DataFrame
        if hasattr(results, 'to_pandas'):
            df = results.to_pandas()
        else:
            df = pd.DataFrame(results)
        
        analysis = {
            "summary": self._calculate_summary_stats(df),
            "per_metric": self._analyze_per_metric(df),
            "threshold_analysis": self._analyze_thresholds(df),
            "correlations": self._calculate_correlations(df),
            "outliers": self._detect_outliers(df)
        }
        
        return analysis
    
    def _calculate_summary_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate summary statistics"""
        metric_cols = [col for col in df.columns if col in self.config.metrics_to_evaluate]
        
        stats = {}
        for metric in metric_cols:
            if metric in df.columns:
                stats[metric] = {
                    "mean": float(df[metric].mean()),
                    "median": float(df[metric].median()),
                    "std": float(df[metric].std()),
                    "min": float(df[metric].min()),
                    "max": float(df[metric].max()),
                    "q25": float(df[metric].quantile(0.25)),
                    "q75": float(df[metric].quantile(0.75))
                }
        
        # Overall score
        if metric_cols:
            overall_mean = df[metric_cols].mean().mean()
            stats["overall"] = {
                "mean": float(overall_mean),
                "std": float(df[metric_cols].mean(axis=1).std())
            }
        
        return stats
    
    def _analyze_per_metric(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detailed per-metric analysis"""
        metric_cols = [col for col in df.columns if col in self.config.metrics_to_evaluate]
        
        analysis = {}
        for metric in metric_cols:
            if metric in df.columns:
                threshold = self.thresholds.get(metric, 0.7)
                above_threshold = (df[metric] >= threshold).sum()
                below_threshold = (df[metric] < threshold).sum()
                
                analysis[metric] = {
                    "above_threshold": int(above_threshold),
                    "below_threshold": int(below_threshold),
                    "pass_rate": float(above_threshold / len(df)),
                    "threshold": threshold
                }
        
        return analysis
    
    def _analyze_thresholds(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze performance against thresholds"""
        metric_cols = [col for col in df.columns if col in self.config.metrics_to_evaluate]
        
        threshold_analysis = {
            "passing_samples": [],
            "failing_samples": []
        }
        
        for idx, row in df.iterrows():
            fails = []
            for metric in metric_cols:
                if metric in row.index:
                    threshold = self.thresholds.get(metric, 0.7)
                    if row[metric] < threshold:
                        fails.append({
                            "metric": metric,
                            "value": float(row[metric]),
                            "threshold": threshold
                        })
            
            if fails:
                threshold_analysis["failing_samples"].append({
                    "index": int(idx),
                    "failures": fails
                })
            else:
                threshold_analysis["passing_samples"].append(int(idx))
        
        return threshold_analysis
    
    def _calculate_correlations(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate correlations between metrics"""
        metric_cols = [col for col in df.columns if col in self.config.metrics_to_evaluate]
        
        if len(metric_cols) < 2:
            return {}
        
        corr_matrix = df[metric_cols].corr()
        
        # Convert to dict
        correlations = {}
        for i, metric1 in enumerate(metric_cols):
            for metric2 in metric_cols[i+1:]:
                key = f"{metric1}_vs_{metric2}"
                correlations[key] = float(corr_matrix.loc[metric1, metric2])
        
        return correlations
    
    def _detect_outliers(self, df: pd.DataFrame) -> Dict[str, List[int]]:
        """Detect outlier samples using IQR method"""
        metric_cols = [col for col in df.columns if col in self.config.metrics_to_evaluate]
        
        outliers = {}
        for metric in metric_cols:
            if metric in df.columns:
                Q1 = df[metric].quantile(0.25)
                Q3 = df[metric].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outlier_indices = df[
                    (df[metric] < lower_bound) | (df[metric] > upper_bound)
                ].index.tolist()
                
                if outlier_indices:
                    outliers[metric] = outlier_indices
        
        return outliers
    
    def display_summary(self, analysis: Dict[str, Any]):
        """Display formatted summary of analysis"""
        print("\n" + "="*70)
        print("EVALUATION SUMMARY")
        print("="*70)
        
        summary = analysis["summary"]
        
        print("\n📊 METRIC STATISTICS")
        print("─"*70)
        
        for metric, stats in summary.items():
            if metric == "overall":
                continue
                
            mean = stats["mean"]
            threshold = self.thresholds.get(metric, 0.7)
            status = self._get_status_emoji(mean, threshold)
            
            print(f"\n{metric.replace('_', ' ').title()}")
            print(f"  Mean:   {mean:.3f} {status}")
            print(f"  Median: {stats['median']:.3f}")
            print(f"  Std:    {stats['std']:.3f}")
            print(f"  Range:  [{stats['min']:.3f}, {stats['max']:.3f}]")
        
        # Overall score
        if "overall" in summary:
            overall = summary["overall"]["mean"]
            overall_threshold = self.thresholds.get("overall", 0.75)
            status = self._get_status_emoji(overall, overall_threshold)
            
            print("\n" + "─"*70)
            print(f"Overall Score: {overall:.3f} {status}")
            print("─"*70)
        
        # Pass rates
        print("\n📈 PASS RATES (Above Threshold)")
        print("─"*70)
        
        per_metric = analysis["per_metric"]
        for metric, data in per_metric.items():
            pass_rate = data["pass_rate"] * 100
            status = "✓" if pass_rate >= 80 else "✗"
            print(f"{metric.replace('_', ' ').title():<25} {pass_rate:>5.1f}% {status}")
        
        # Correlations
        if analysis["correlations"]:
            print("\n🔗 METRIC CORRELATIONS")
            print("─"*70)
            for pair, corr in analysis["correlations"].items():
                print(f"{pair.replace('_', ' ').title():<45} {corr:>6.3f}")
        
        # Outliers
        if analysis["outliers"]:
            print("\n⚠️  OUTLIERS DETECTED")
            print("─"*70)
            for metric, indices in analysis["outliers"].items():
                print(f"{metric}: {len(indices)} samples - indices: {indices[:5]}...")
    
    def _get_status_emoji(self, value: float, threshold: float) -> str:
        """Get status emoji based on value and threshold"""
        if value >= threshold + 0.1:
            return "🟢 EXCELLENT"
        elif value >= threshold:
            return "🟡 GOOD"
        elif value >= threshold - 0.1:
            return "🟠 ACCEPTABLE"
        else:
            return "🔴 NEEDS IMPROVEMENT"
    
    def export_results(
        self,
        analysis: Dict[str, Any],
        dataset: List[Dict[str, Any]],
        output_dir: Optional[str] = None
    ):
        """Export results in multiple formats"""
        output_dir = output_dir or self.config.results_dir
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        
        # JSON export
        if "json" in self.config.export_formats:
            json_file = output_path / f"evaluation_results_{timestamp}.json"
            with open(json_file, 'w') as f:
                json.dump({
                    "analysis": analysis,
                    "dataset": dataset
                }, f, indent=2, default=str)
            print(f"✓ Exported JSON: {json_file}")
        
        # CSV export
        if "csv" in self.config.export_formats:
            csv_file = output_path / f"evaluation_results_{timestamp}.csv"
            
            # Flatten data for CSV
            rows = []
            for item in dataset:
                row = {
                    "question": item["question"],
                    "answer": item["answer"],
                    "ground_truth": item["ground_truth"],
                    "num_contexts": len(item["contexts"])
                }
                rows.append(row)
            
            df = pd.DataFrame(rows)
            df.to_csv(csv_file, index=False)
            print(f"✓ Exported CSV: {csv_file}")
        
        # HTML report
        if "html" in self.config.export_formats:
            html_file = output_path / f"evaluation_report_{timestamp}.html"
            self._generate_html_report(analysis, dataset, html_file)
            print(f"✓ Exported HTML: {html_file}")
    
    def _generate_html_report(
        self,
        analysis: Dict[str, Any],
        dataset: List[Dict[str, Any]],
        output_file: Path
    ):
        """Generate HTML report"""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>RAG Evaluation Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1 {{ color: #333; }}
                h2 {{ color: #666; border-bottom: 2px solid #ddd; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                th {{ background-color: #4CAF50; color: white; }}
                .metric-box {{ 
                    display: inline-block; 
                    margin: 10px; 
                    padding: 20px; 
                    border-radius: 8px; 
                    background-color: #f5f5f5; 
                }}
                .excellent {{ background-color: #d4edda; }}
                .good {{ background-color: #fff3cd; }}
                .poor {{ background-color: #f8d7da; }}
            </style>
        </head>
        <body>
            <h1>🎯 RAG System Evaluation Report</h1>
            <p><strong>Generated:</strong> {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            <p><strong>Total Samples:</strong> {len(dataset)}</p>
            
            <h2>📊 Summary Statistics</h2>
        """
        
        summary = analysis["summary"]
        for metric, stats in summary.items():
            if metric == "overall":
                continue
            mean = stats["mean"]
            css_class = "excellent" if mean >= 0.8 else "good" if mean >= 0.7 else "poor"
            html_content += f"""
            <div class="metric-box {css_class}">
                <h3>{metric.replace('_', ' ').title()}</h3>
                <p><strong>Mean:</strong> {mean:.3f}</p>
                <p><strong>Std:</strong> {stats['std']:.3f}</p>
            </div>
            """
        
        html_content += """
        </body>
        </html>
        """
        
        with open(output_file, 'w') as f:
            f.write(html_content)