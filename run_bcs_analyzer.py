import os
from bcs_analyzer import BCSQuestionAnalyzer
from bcs_usage_guide import BCSDataProcessor, AdvancedBCSAnalyzer
import re
def validate_tsv_format(content):
    """Validate if the content follows proper TSV format for BCS questions"""
    lines = content.strip().split('\n')
    for line in lines:
        if not line.strip() or line.startswith('#'):
            continue
        parts = line.split('\t')
        if len(parts) < 7:  # Minimum: question_id, question, 4 options, correct_answer
            return False
    return True

def main():
    # Sample BCS question data in TSV format (tab-separated values)
    sample_data = {
        42: """1	What is the correct indirect form of: He said, "You had better see a doctor"	He advised him to see a doctor.	He advised that he should see a doctor.	He suggested that he seen a doctor.	He proposed to see a doctor.	He advised him to see a doctor.	Direct to indirect speech conversion rule.	BCS_42 BCS_ENG""",
        41: """2	Identify the word that remains the same in plural form.	deer	horse	elephant	tiger	deer	Some words have same singular and plural forms.	BCS_41 BCS_ENG""",
        40: """3	Which word is correct?	Furnitures	Informations	Sceneries	Proceeds	Proceeds	Non-count Noun plural rule.	BCS_40 BCS_ENG"""
    }

    print("Starting BCS Question Analysis...")

    # Validate TSV format
    for year, content in sample_data.items():
        if not validate_tsv_format(content):
            print(f"Error: Invalid TSV format in year {year} data!")
            return

    try:
        # Create an instance of the advanced analyzer
        analyzer = AdvancedBCSAnalyzer()

        # Run the analysis
        print("\nRunning comprehensive analysis...")
        results = analyzer.run_full_analysis(sample_data)

        # Generate study recommendations
        print("\nGenerating study recommendations...")
        recommendations = analyzer.generate_study_recommendations()

        # Create a mock exam
        print("\nCreating a mock exam...")
        mock_exam = analyzer.generate_mock_exam(num_questions=10)

        # Export comprehensive report
        print("\nExporting analysis reports...")
        report_files = analyzer.export_comprehensive_report()

        print("\nAnalysis complete! Check the generated files in your directory.")
    except Exception as e:
        print(f"\nError during analysis: {str(e)}")
        return

if __name__ == "__main__":
    main()