# BCS Question Analysis System - Complete Usage Guide
# =====================================================

import os
import json
from datetime import datetime, timedelta
import pickle

class BCSDataProcessor:
    """Helper class to process multiple BCS files"""
    
    def __init__(self):
        self.file_contents = {}
        self.metadata = {}
    
    def load_multiple_files(self, file_paths):
        """Load multiple BCS text files"""
        for file_path in file_paths:
            try:
                year = self.extract_year_from_filename(file_path)
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    self.file_contents[year] = content
                    self.metadata[year] = {
                        'file_path': file_path,
                        'loaded_at': datetime.now(),
                        'size': len(content)
                    }
                print(f"Loaded BCS {year}: {len(content)} characters")
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        
        return self.file_contents
    
    def extract_year_from_filename(self, filename):
        """Extract BCS year from filename"""
        import re
        match = re.search(r'(?:BCS_?|bcs_?)(\d{2})', filename.lower())
        if match:
            year = int(match.group(1))
            # Convert 2-digit to 4-digit year (assuming 21st century)
            if year < 50:
                return 2000 + year
            else:
                return 1900 + year
        return datetime.now().year
    
    def validate_file_format(self, content):
        """Validate if file follows expected BCS format"""
        lines = content.strip().split('\n')
        valid_lines = 0
        
        for line in lines:
            if line.strip() and not line.startswith('#'):
                parts = line.split('\t')
                if len(parts) >= 7:
                    valid_lines += 1
        
        return valid_lines > 0, valid_lines

class AdvancedBCSAnalyzer(BCSQuestionAnalyzer):
    """Extended analyzer with advanced features"""
    
    def __init__(self):
        super().__init__()
        self.difficulty_levels = {}
        self.topic_clusters = {}
        self.temporal_patterns = {}
    
    def analyze_difficulty_progression(self):
        """Analyze how question difficulty changes over years"""
        print("Analyzing difficulty progression...")
        
        for year in sorted(self.df['year'].unique()):
            year_data = self.df[self.df['year'] == year]
            
            # Calculate difficulty indicators
            avg_question_length = year_data['question'].str.len().mean()
            complex_questions = year_data['question'].str.contains(
                r'[A-Z]{3,}|[০-৯]+|[0-9]{4,}', na=False).sum()
            
            difficulty_score = (avg_question_length / 100) + (complex_questions / len(year_data))
            
            self.difficulty_levels[year] = {
                'score': difficulty_score,
                'avg_length': avg_question_length,
                'complex_count': complex_questions,
                'total_questions': len(year_data)
            }
        
        return self.difficulty_levels
    
    def cluster_topics(self, n_clusters=8):
        """Cluster questions by topics using ML"""
        print(f"Clustering questions into {n_clusters} topic groups...")
        
        # Prepare text data
        questions = self.df['question'].fillna('').tolist()
        
        # Vectorize questions
        vectorizer = TfidfVectorizer(
            max_features=500, 
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        X = vectorizer.fit_transform(questions)
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(X)
        
        # Analyze clusters
        feature_names = vectorizer.get_feature_names_out()
        
        for i in range(n_clusters):
            cluster_questions = self.df[clusters == i]
            
            # Get top terms for this cluster
            cluster_center = kmeans.cluster_centers_[i]
            top_indices = cluster_center.argsort()[-10:][::-1]
            top_terms = [feature_names[idx] for idx in top_indices]
            
            self.topic_clusters[f"Cluster_{i}"] = {
                'size': len(cluster_questions),
                'top_terms': top_terms,
                'subjects': cluster_questions['subject'].value_counts().to_dict(),
                'years': cluster_questions['year'].value_counts().to_dict(),
                'sample_questions': cluster_questions['question'].head(3).tolist()
            }
        
        return self.topic_clusters
    
    def analyze_temporal_patterns(self):
        """Analyze how topics change over time"""
        print("Analyzing temporal patterns...")
        
        # Track topic evolution
        years = sorted(self.df['year'].unique())
        
        for subject in self.subjects.keys():
            subject_data = self.df[self.df['subject'] == subject]
            year_evolution = {}
            
            for year in years:
                year_subject_data = subject_data[subject_data['year'] == year]
                if len(year_subject_data) > 0:
                    # Extract key terms for this year
                    questions_text = ' '.join(year_subject_data['question'].fillna(''))
                    tokens = word_tokenize(questions_text.lower())
                    tokens = [token for token in tokens if token.isalpha() and len(token) > 3]
                    
                    year_evolution[year] = Counter(tokens).most_common(10)
            
            self.temporal_patterns[subject] = year_evolution
        
        return self.temporal_patterns
    
    def predict_exam_difficulty(self, next_year=None):
        """Predict difficulty level of next exam"""
        if not next_year:
            next_year = max(self.df['year']) + 1
        
        # Analyze difficulty trend
        years = sorted(self.difficulty_levels.keys())[-5:]  # Last 5 years
        difficulty_scores = [self.difficulty_levels[year]['score'] for year in years]
        
        # Simple trend analysis
        if len(difficulty_scores) >= 3:
            recent_trend = np.mean(difficulty_scores[-2:]) - np.mean(difficulty_scores[:-2])
            predicted_difficulty = difficulty_scores[-1] + recent_trend
        else:
            predicted_difficulty = np.mean(difficulty_scores)
        
        # Classify difficulty level
        if predicted_difficulty < 1.0:
            level = "Easy"
        elif predicted_difficulty < 1.5:
            level = "Moderate"
        elif predicted_difficulty < 2.0:
            level = "Difficult"
        else:
            level = "Very Difficult"
        
        return {
            'predicted_year': next_year,
            'difficulty_score': predicted_difficulty,
            'difficulty_level': level,
            'confidence': min(0.9, len(difficulty_scores) / 5.0),
            'trend': 'Increasing' if recent_trend > 0.1 else 'Decreasing' if recent_trend < -0.1 else 'Stable'
        }
    
    def generate_personalized_study_plan(self, weak_subjects=None, available_days=30):
        """Generate a personalized study plan"""
        print("Generating personalized study plan...")
        
        study_plan = {
            'duration_days': available_days,
            'daily_schedule': {},
            'weekly_targets': {},
            'subject_allocation': {}
        }
        
        # Calculate time allocation based on subject priority
        total_priority = sum(data['priority_score'] for data in self.subject_weights.values())
        
        for subject, data in self.subject_weights.items():
            allocation_ratio = data['priority_score'] / total_priority
            
            # Increase allocation for weak subjects
            if weak_subjects and subject in weak_subjects:
                allocation_ratio *= 1.5
            
            days_allocated = max(2, int(available_days * allocation_ratio))
            study_plan['subject_allocation'][subject] = {
                'days': days_allocated,
                'priority': data['priority_score'],
                'focus_areas': self._get_subject_focus_areas(subject),
                'daily_hours': 2 if subject in (weak_subjects or []) else 1.5
            }
        
        # Create weekly targets
        weeks = available_days // 7
        for week in range(1, weeks + 1):
            study_plan['weekly_targets'][f'Week_{week}'] = self._generate_weekly_targets(week)
        
        return study_plan
    
    def _get_subject_focus_areas(self, subject):
        """Get focus areas for a specific subject"""
        focus_areas = {
            'English': ['Grammar', 'Vocabulary', 'Sentence Structure', 'Literature'],
            'Bangla': ['ব্যাকরণ', 'সাহিত্য', 'বানান', 'প্রবাদ-প্রবচন'],
            'Mathematics': ['Arithmetic', 'Algebra', 'Geometry', 'Statistics'],
            'General Knowledge': ['Bangladesh Affairs', 'Current Events', 'Geography', 'History'],
            'International Affairs': ['World Politics', 'Organizations', 'Treaties', 'Current Issues']
        }
        return focus_areas.get(subject, ['General Topics'])
    
    def _generate_weekly_targets(self, week_number):
        """Generate targets for a specific week"""
        base_targets = {
            'questions_to_solve': 50 + (week_number * 10),
            'topics_to_cover': 5 + week_number,
            'mock_tests': 1 if week_number % 2 == 0 else 0,
            'revision_hours': 2 * week_number
        }
        return base_targets
    
    def generate_mock_exam(self, num_questions=100, subject_distribution=None):
        """Generate a mock exam based on patterns"""
        print(f"Generating mock exam with {num_questions} questions...")
        
        if not subject_distribution:
            # Default distribution based on actual BCS pattern
            subject_distribution = {
                'Bangla': 35,
                'English': 35,
                'Mathematics': 15,
                'General Knowledge': 10,
                'International Affairs': 5
            }
        
        mock_exam = {
            'exam_info': {
                'total_questions': num_questions,
                'time_limit': '3 hours',
                'generated_on': datetime.now().isoformat(),
                'difficulty_level': 'Mixed'
            },
            'questions': [],
            'answer_key': {},
            'subject_breakdown': subject_distribution
        }
        
        question_id = 1
        
        for subject, count in subject_distribution.items():
            if subject in self.subjects:
                # Get sample questions from this subject
                subject_questions = self.df[self.df['subject'] == subject].sample(
                    min(count, len(self.subjects[subject]))
                )
                
                for _, row in subject_questions.iterrows():
                    mock_question = {
                        'id': question_id,
                        'subject': subject,
                        'question': row['question'],
                        'options': {
                            'A': row['option_a'],
                            'B': row['option_b'],
                            'C': row['option_c'],
                            'D': row['option_d']
                        },
                        'difficulty': self._estimate_question_difficulty(row['question'])
                    }
                    
                    mock_exam['questions'].append(mock_question)
                    mock_exam['answer_key'][question_id] = row['correct_answer']
                    question_id += 1
        
        return mock_exam
    
    def _estimate_question_difficulty(self, question):
        """Estimate individual question difficulty"""
        length_score = len(question) / 100
        complexity_score = len(re.findall(r'[A-Z]{2,}|[০-৯]+|[0-9]{3,}', question)) * 0.2
        total_score = length_score + complexity_score
        
        if total_score < 0.8:
            return 'Easy'
        elif total_score < 1.5:
            return 'Medium'
        else:
            return 'Hard'
    
    def export_comprehensive_report(self, filename_prefix='bcs_analysis'):
        """Export comprehensive analysis report"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Main report
        report_filename = f"{filename_prefix}_{timestamp}.html"
        
        html_content = self._generate_html_report()
        
        with open(report_filename, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        # Export data as JSON
        json_filename = f"{filename_prefix}_data_{timestamp}.json"
        export_data = {
            'metadata': {
                'generated_on': datetime.now().isoformat(),
                'total_questions': len(self.df),
                'subjects': list(self.subjects.keys()),
                'years_covered': sorted(self.df['year'].unique().tolist())
            },
            'subject_weights': self.subject_weights,
            'difficulty_levels': self.difficulty_levels,
            'topic_clusters': self.topic_clusters,
            'question_patterns': self.question_patterns,
            'predictions': self.predict_next_questions(30)
        }
        
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        # Export study plan
        study_plan = self.generate_personalized_study_plan()
        plan_filename = f"study_plan_{timestamp}.json"
        
        with open(plan_filename, 'w', encoding='utf-8') as f:
            json.dump(study_plan, f, indent=2, ensure_ascii=False)
        
        print(f"Reports exported:")
        print(f"  - HTML Report: {report_filename}")
        print(f"  - Data Export: {json_filename}")
        print(f"  - Study Plan: {plan_filename}")
        
        return report_filename, json_filename, plan_filename
    
    def _generate_html_report(self):
        """Generate comprehensive HTML report"""
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>BCS Question Analysis Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }}
        .container {{ background-color: white; padding: 30px; border-radius: 10px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
        .header {{ background-color: #2c3e50; color: white; padding: 20px; border-radius: 5px; text-align: center; }}
        .section {{ margin: 30px 0; padding: 20px; border-left: 4px solid #3498db; background-color: #ecf0f1; }}
        .metric {{ display: inline-block; margin: 15px; padding: 15px; background-color: #3498db; color: white; border-radius: 5px; text-align: center; }}
        .prediction {{ background-color: #e8f5e8; padding: 10px; margin: 5px 0; border-radius: 5px; border-left: 3px solid #27ae60; }}
        table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #34495e; color: white; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
        .high-priority {{ background-color: #e74c3c; color: white; }}
        .medium-priority {{ background-color: #f39c12; color: white; }}
        .low-priority {{ background-color: #27ae60; color: white; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>BCS Question Analysis Report</h1>
            <p>Generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</p>
        </div>
        
        <div class="section">
            <h2>📊 Summary Statistics</h2>
            <div class="metric">
                <h3>{len(self.df)}</h3>
                <p>Total Questions</p>
            </div>
            <div class="metric">
                <h3>{len(self.subjects)}</h3>
                <p>Subjects</p>
            </div>
            <div class="metric">
                <h3>{len(self.df['year'].unique())}</h3>
                <p>Years Covered</p>
            </div>
            <div class="metric">
                <h3>{len(self.topic_clusters) if self.topic_clusters else 0}</h3>
                <p>Topic Clusters</p>
            </div>
        </div>
        
        <div class="section">
            <h2>🎯 Subject Priority Analysis</h2>
            <table>
                <tr>
                    <th>Subject</th>
                    <th>Questions</th>
                    <th>Importance</th>
                    <th>Priority Score</th>
                    <th>Recommendation</th>
                </tr>
        """
        
        # Add subject priority table
        for subject, data in sorted(self.subject_weights.items(), 
                                  key=lambda x: x[1]['priority_score'], 
                                  reverse=True):
            priority_class = ('high-priority' if data['priority_score'] > 0.8 
                            else 'medium-priority' if data['priority_score'] > 0.5 
                            else 'low-priority')
            
            html += f"""
                <tr>
                    <td><strong>{subject}</strong></td>
                    <td>{data['frequency']}</td>
                    <td>{data['importance']:.3f}</td>
                    <td class="{priority_class}">{data['priority_score']:.3f}</td>
                    <td>{'High Priority' if data['priority_score'] > 0.8 else 'Medium Priority' if data['priority_score'] > 0.5 else 'Low Priority'}</td>
                </tr>
            """
        
        html += """
            </table>
        </div>
        
        <div class="section">
            <h2>🔮 Predicted Questions for Next Exam</h2>
        """
        
        # Add predictions
        predictions = self.predict_next_questions(15)
        for i, pred in enumerate(predictions, 1):
            html += f"""
            <div class="prediction">
                <strong>{i}. [{pred['subject']}]</strong> {pred['predicted_question']}<br>
                <small>Confidence: {pred['confidence']:.2f}</small>
            </div>
            """
        
        html += """
        </div>
        
        <div class="section">
            <h2>📚 Study Recommendations</h2>
        """
        
        # Add study recommendations
        recommendations = self.generate_study_recommendations()
        
        html += "<h3>High Priority Subjects:</h3><ul>"
        for rec in recommendations['high_priority_subjects']:
            html += f"<li><strong>{rec['subject']}</strong> - {rec['reason']}</li>"
        html += "</ul>"
        
        html += "<h3>Important Topics to Focus:</h3><ul>"
        for topic in recommendations['topics_to_focus'][:10]:
            html += f"<li><strong>{topic['topic']}</strong> (appeared {topic['frequency']} times)</li>"
        html += "</ul>"
        
        html += """
        </div>
        
        <div class="section">
            <h2>📈 Question Pattern Analysis</h2>
            <table>
                <tr>
                    <th>Pattern Type</th>
                    <th>Frequency</th>
                    <th>Percentage</th>
                </tr>
        """
        
        total_patterns = sum(self.question_patterns.values())
        for pattern, count in sorted(self.question_patterns.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_patterns * 100) if total_patterns > 0 else 0
            html += f"""
                <tr>
                    <td>{pattern.replace('_', ' ').title()}</td>
                    <td>{count}</td>
                    <td>{percentage:.1f}%</td>
                </tr>
            """
        
        html += """
            </table>
        </div>
        
        <div class="section">
            <h2>💡 Tips for Success</h2>
            <ul>
                <li><strong>Focus on High Priority Subjects:</strong> Allocate more study time to subjects with higher priority scores</li>
                <li><strong>Practice Question Types:</strong> Master the most frequent question patterns identified in the analysis</li>
                <li><strong>Regular Mock Tests:</strong> Take practice tests to simulate exam conditions</li>
                <li><strong>Track Progress:</strong> Monitor your performance in weak areas identified by the analysis</li>
                <li><strong>Stay Updated:</strong> Keep track of current affairs, especially for General Knowledge sections</li>
            </ul>
        </div>
        
        <div class="section">
            <h2>📊 Data Sources</h2>
            <p>This analysis is based on questions from BCS examinations spanning multiple years. 
            The predictions and recommendations are generated using machine learning algorithms 
            that analyze patterns in question types, subjects, and difficulty levels.</p>
            <p><strong>Disclaimer:</strong> These predictions are based on historical patterns and should 
            be used as a study guide. Actual exam questions may vary.</p>
        </div>
    </div>
</body>
</html>
        """
        
        return html

# Complete Usage Example and Tutorial
def complete_usage_example():
    """
    Complete example of how to use the BCS Analysis System
    """
    print("BCS Question Analysis System - Complete Usage Guide")
    print("=" * 60)
    
    # Step 1: Initialize the system
    print("\n1. Initializing the system...")
    processor = BCSDataProcessor()
    analyzer = AdvancedBCSAnalyzer()
    
    # Step 2: Load your BCS files
    print("\n2. Loading BCS question files...")
    
    # Example file paths - replace with your actual file paths
    file_paths = [
        "BCS_42.txt",
        "BCS_41.txt", 
        "BCS_40.txt",
        "BCS_39.txt",
        "BCS_38.txt"
    ]
    
    # In practice, you would use actual files:
    # file_contents = processor.load_multiple_files(file_paths)
    
    # For demo, using sample data
    sample_data = {
        42: """1. What is the correct indirect form of: He said, "You had better see a doctor"	He advised him to see a doctor.	He advised that he should see a doctor.	He suggested that he seen a doctor.	He proposed to see a doctor.	He advised him to see a doctor.	Direct to indirect speech conversion rule.	BCS_42 BCS_ENG
2. Identify the word that remains the same in plural form.	deer	horse	elephant	tiger	deer	Some words have same singular and plural forms.	BCS_42 BCS_ENG""",
        41: """3. Fill in the blank: I ____ him yesterday.	see	saw	seen	seeing	saw	Past tense usage.	BCS_41 BCS_ENG
4. গণপ্রজাতন্ত্রী বাংলাদেশের সংবিধানের খসড়া সর্বপ্রথম গণপরিষদে ১৯৭২ সালের কোন তারিখে উত্থাপিত হয়?	১১ নভেম্বর	১২ অক্টোবর	১৬ ডিসেম্বর	৩ মার্চ	১২ অক্টোবর	Constitutional history of Bangladesh.	BCS_41 BCS_GK""",
        40: """5. Solve: 2x + 3 = 11	x = 4	x = 5	x = 6	x = 7	x = 4	Basic algebra.	BCS_40 BCS_MTH
6. What is the capital of Australia?	Sydney	Melbourne	Canberra	Perth	Canberra	World geography.	BCS_40 BCS_GK_INT"""
    }
    
    print("   Sample data loaded for demonstration")
    
    # Step 3: Run comprehensive analysis
    print("\n3. Running comprehensive analysis...")
    results = analyzer.run_full_analysis(sample_data)
    
    # Step 4: Advanced analysis features
    print("\n4. Running advanced analysis...")
    
    # Difficulty analysis
    difficulty_analysis = analyzer.analyze_difficulty_progression()
    print(f"   Difficulty levels analyzed for {len(difficulty_analysis)} years")
    
    # Topic clustering
    clusters = analyzer.cluster_topics(n_clusters=5)
    print(f"   Questions clustered into {len(clusters)} topic groups")
    
    # Temporal patterns
    temporal = analyzer.analyze_temporal_patterns()
    print(f"   Temporal patterns analyzed for {len(temporal)} subjects")
    
    # Step 5: Generate personalized study plan
    print("\n5. Generating personalized study plan...")
    study_plan = analyzer.generate_personalized_study_plan(
        weak_subjects=['Mathematics', 'English'],
        available_days=45
    )
    print(f"   Study plan generated for {study_plan['duration_days']} days")
    
    # Step 6: Create mock exam
    print("\n6. Generating mock exam...")
    mock_exam = analyzer.generate_mock_exam(num_questions=20)
    print(f"   Mock exam created with {len(mock_exam['questions'])} questions")
    
    # Step 7: Export comprehensive report
    print("\n7. Exporting comprehensive reports...")
    try:
        report_files = analyzer.export_comprehensive_report()
        print(f"   Reports exported successfully: {len(report_files)} files")
    except Exception as e:
        print(f"   Export simulation completed (actual files not created in demo)")
    
    # Step 8: Display key insights
    print("\n8. Key Insights from Analysis:")
    print("   " + "="*40)
    
    print("\n   📊 Subject Priorities:")
    for subject, data in list(results['subject_weights'].items())[:3]:
        print(f"      {subject}: {data['priority_score']:.3f} priority score")
    
    print(f"\n   🎯 Top Predictions (sample):")
    for i, pred in enumerate(results['predictions'][:3], 1):
        print(f"      {i}. [{pred['subject']}] {pred['predicted_question'][:60]}...")
    
    print("\n   💡 Study Recommendations:")
    recs = results['recommendations']
    for rec in recs['high_priority_subjects'][:2]:
        print(f"      Focus on {rec['subject']} - {rec['reason']}")
    
    print("\n9. Next Steps:")
    print("   =" + "="*30)
    print("   - Replace sample data with your actual BCS files")
    print("   - Adjust analysis parameters based on your needs")
    print("   - Use generated study plan for exam preparation")
    print("   - Regularly update data with new BCS questions")
    print("   - Practice with generated mock exams")

if __name__ == "__main__":
    complete_usage_example()