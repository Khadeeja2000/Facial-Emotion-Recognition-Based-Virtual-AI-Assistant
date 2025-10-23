"""
Complete Ensemble Learning Pipeline for Emotion Recognition
Orchestrates the entire ensemble learning process from training to real-world testing
"""

import os
import sys
import time
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

def print_header(title: str):
    """Print formatted header"""
    print("\n" + "="*80)
    print(f" {title}")
    print("="*80)

def print_step(step_num: int, total_steps: int, description: str):
    """Print formatted step"""
    print(f"\n[STEP {step_num}/{total_steps}] {description}")
    print("-" * 60)

def main():
    """Main ensemble learning pipeline"""
    
    print_header("ENSEMBLE LEARNING PIPELINE FOR EMOTION RECOGNITION")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Pipeline steps
    steps = [
        ("Training Individual Models", "ensemble_trainer.py"),
        ("Comprehensive Evaluation", "ensemble_evaluator.py"),
        ("Model Comparison Analysis", "ensemble_comparison.py"),
        ("Real-World Testing", "real_world_testing.py")
    ]
    
    total_steps = len(steps)
    results_path = Path("ensemble_learning/results")
    results_path.mkdir(parents=True, exist_ok=True)
    
    # Track pipeline execution
    pipeline_log = {
        "start_time": datetime.now().isoformat(),
        "steps_completed": [],
        "steps_failed": [],
        "overall_success": False
    }
    
    try:
        for step_num, (description, script_name) in enumerate(steps, 1):
            print_step(step_num, total_steps, description)
            
            script_path = Path(__file__).parent / script_name
            
            if not script_path.exists():
                print(f"✗ Script not found: {script_name}")
                pipeline_log["steps_failed"].append({
                    "step": step_num,
                    "description": description,
                    "script": script_name,
                    "error": "Script not found"
                })
                continue
            
            print(f"Executing: {script_name}")
            start_time = time.time()
            
            try:
                # Import and run the script
                if script_name == "ensemble_trainer.py":
                    from ensemble_trainer import main as trainer_main
                    trainer_main()
                elif script_name == "ensemble_evaluator.py":
                    from ensemble_evaluator import main as evaluator_main
                    evaluator_main()
                elif script_name == "ensemble_comparison.py":
                    from ensemble_comparison import main as comparison_main
                    comparison_main()
                elif script_name == "real_world_testing.py":
                    from real_world_testing import main as testing_main
                    testing_main()
                
                execution_time = time.time() - start_time
                print(f"✓ Completed in {execution_time:.2f} seconds")
                
                pipeline_log["steps_completed"].append({
                    "step": step_num,
                    "description": description,
                    "script": script_name,
                    "execution_time": execution_time
                })
                
            except Exception as e:
                execution_time = time.time() - start_time
                print(f"✗ Failed after {execution_time:.2f} seconds")
                print(f"Error: {str(e)}")
                
                pipeline_log["steps_failed"].append({
                    "step": step_num,
                    "description": description,
                    "script": script_name,
                    "error": str(e),
                    "execution_time": execution_time
                })
                
                # Ask user if they want to continue
                response = input("\nDo you want to continue with the next step? (y/n): ").lower()
                if response != 'y':
                    print("Pipeline stopped by user")
                    break
        
        # Check overall success
        if len(pipeline_log["steps_completed"]) == total_steps:
            pipeline_log["overall_success"] = True
            print_header("PIPELINE COMPLETED SUCCESSFULLY")
        else:
            print_header("PIPELINE COMPLETED WITH SOME FAILURES")
        
        # Print summary
        print(f"\nSUMMARY:")
        print(f"Steps completed: {len(pipeline_log['steps_completed'])}/{total_steps}")
        print(f"Steps failed: {len(pipeline_log['steps_failed'])}")
        
        if pipeline_log["steps_completed"]:
            print(f"\n✓ Completed steps:")
            for step in pipeline_log["steps_completed"]:
                print(f"  - {step['description']} ({step['execution_time']:.2f}s)")
        
        if pipeline_log["steps_failed"]:
            print(f"\n✗ Failed steps:")
            for step in pipeline_log["steps_failed"]:
                print(f"  - {step['description']}: {step['error']}")
        
        # Save pipeline log
        import json
        pipeline_log["end_time"] = datetime.now().isoformat()
        log_path = results_path / "pipeline_execution_log.json"
        
        with open(log_path, 'w') as f:
            json.dump(pipeline_log, f, indent=2)
        
        print(f"\nPipeline log saved to: {log_path}")
        
        # Generate final report
        generate_final_report(pipeline_log, results_path)
        
    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user")
        pipeline_log["end_time"] = datetime.now().isoformat()
        pipeline_log["interrupted"] = True
        
        # Save partial log
        import json
        log_path = results_path / "pipeline_execution_log.json"
        with open(log_path, 'w') as f:
            json.dump(pipeline_log, f, indent=2)
        
        print(f"Partial pipeline log saved to: {log_path}")
    
    except Exception as e:
        print(f"\nUnexpected error in pipeline: {str(e)}")
        pipeline_log["end_time"] = datetime.now().isoformat()
        pipeline_log["unexpected_error"] = str(e)
        
        # Save error log
        import json
        log_path = results_path / "pipeline_execution_log.json"
        with open(log_path, 'w') as f:
            json.dump(pipeline_log, f, indent=2)

def generate_final_report(pipeline_log: dict, results_path: Path):
    """Generate final pipeline report"""
    
    print("\nGenerating final pipeline report...")
    
    report_path = results_path / "FINAL_ENSEMBLE_REPORT.txt"
    
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("FINAL ENSEMBLE LEARNING PIPELINE REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("PIPELINE EXECUTION SUMMARY\n")
        f.write("-" * 30 + "\n")
        f.write(f"Start Time: {pipeline_log['start_time']}\n")
        f.write(f"End Time: {pipeline_log.get('end_time', 'N/A')}\n")
        f.write(f"Overall Success: {'✓ YES' if pipeline_log['overall_success'] else '✗ NO'}\n")
        f.write(f"Steps Completed: {len(pipeline_log['steps_completed'])}\n")
        f.write(f"Steps Failed: {len(pipeline_log['steps_failed'])}\n")
        
        if pipeline_log.get('interrupted'):
            f.write("Pipeline Status: INTERRUPTED BY USER\n")
        elif pipeline_log.get('unexpected_error'):
            f.write(f"Pipeline Status: FAILED - {pipeline_log['unexpected_error']}\n")
        elif pipeline_log['overall_success']:
            f.write("Pipeline Status: COMPLETED SUCCESSFULLY\n")
        else:
            f.write("Pipeline Status: COMPLETED WITH FAILURES\n")
        
        f.write("\n\nDETAILED STEP RESULTS\n")
        f.write("-" * 25 + "\n")
        
        for step in pipeline_log['steps_completed']:
            f.write(f"\n✓ STEP {step['step']}: {step['description']}\n")
            f.write(f"  Script: {step['script']}\n")
            f.write(f"  Execution Time: {step['execution_time']:.2f} seconds\n")
            f.write(f"  Status: SUCCESS\n")
        
        for step in pipeline_log['steps_failed']:
            f.write(f"\n✗ STEP {step['step']}: {step['description']}\n")
            f.write(f"  Script: {step['script']}\n")
            f.write(f"  Execution Time: {step.get('execution_time', 0):.2f} seconds\n")
            f.write(f"  Error: {step['error']}\n")
            f.write(f"  Status: FAILED\n")
        
        f.write("\n\nGENERATED OUTPUTS\n")
        f.write("-" * 20 + "\n")
        
        # List generated files
        output_files = [
            "mini_xception_ensemble.h5",
            "mobilenetv2_ensemble.h5", 
            "efficientnetb0_ensemble.h5",
            "ensemble_weights.json",
            "evaluation_results.json",
            "comprehensive_evaluation_report.txt",
            "ensemble_comparison_report.txt",
            "real_world_testing_report.txt",
            "comprehensive_comparison.png",
            "per_class_performance_comparison.png",
            "improvement_analysis.png",
            "confusion_matrix_comparison.png",
            "performance_radar_chart.png",
            "ensemble_contribution_analysis.png"
        ]
        
        for filename in output_files:
            file_path = results_path / filename
            if file_path.exists():
                f.write(f"✓ {filename}\n")
            else:
                f.write(f"✗ {filename} (not generated)\n")
        
        f.write("\n\nRECOMMENDATIONS\n")
        f.write("-" * 15 + "\n")
        
        if pipeline_log['overall_success']:
            f.write("✓ Ensemble learning pipeline completed successfully!\n")
            f.write("✓ All models trained and evaluated\n")
            f.write("✓ Comprehensive performance analysis completed\n")
            f.write("✓ Real-world testing performed\n")
            f.write("\nNext steps:\n")
            f.write("1. Review the generated reports and visualizations\n")
            f.write("2. Analyze the ensemble performance improvements\n")
            f.write("3. Deploy the ensemble model for production use\n")
            f.write("4. Consider further optimization based on results\n")
        else:
            f.write("⚠ Pipeline completed with some failures\n")
            f.write("\nRecommended actions:\n")
            f.write("1. Review failed steps and error messages\n")
            f.write("2. Fix any data or configuration issues\n")
            f.write("3. Re-run the pipeline or individual scripts\n")
            f.write("4. Check system requirements and dependencies\n")
        
        f.write("\n\nTECHNICAL NOTES\n")
        f.write("-" * 15 + "\n")
        f.write("• Ensemble combines Mini-XCEPTION, MobileNetV2, and EfficientNetB0\n")
        f.write("• Weighted voting strategy used for ensemble predictions\n")
        f.write("• All models evaluated on FER2013 test dataset\n")
        f.write("• Real-world testing performed with live camera feed\n")
        f.write("• Comprehensive metrics calculated for all models\n")
    
    print(f"Final report saved to: {report_path}")
    
    # Print summary to console
    print(f"\n{'='*60}")
    print("FINAL PIPELINE SUMMARY")
    print(f"{'='*60}")
    
    if pipeline_log['overall_success']:
        print("🎉 ENSEMBLE LEARNING PIPELINE COMPLETED SUCCESSFULLY!")
        print("\nGenerated outputs:")
        print("• Trained ensemble models")
        print("• Comprehensive evaluation results")
        print("• Performance comparison visualizations")
        print("• Real-world testing analysis")
        print("• Detailed reports and documentation")
        
        print(f"\n📊 Key files to review:")
        print(f"• {results_path}/comprehensive_evaluation_report.txt")
        print(f"• {results_path}/ensemble_comparison_report.txt")
        print(f"• {results_path}/FINAL_ENSEMBLE_REPORT.txt")
        
    else:
        print("⚠ PIPELINE COMPLETED WITH ISSUES")
        print(f"\nCompleted: {len(pipeline_log['steps_completed'])} steps")
        print(f"Failed: {len(pipeline_log['steps_failed'])} steps")
        
        if pipeline_log['steps_failed']:
            print("\nFailed steps:")
            for step in pipeline_log['steps_failed']:
                print(f"• {step['description']}: {step['error']}")

if __name__ == "__main__":
    main()
