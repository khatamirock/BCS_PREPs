import pandas as pd
import numpy as np
import re
from collections import defaultdict, Counter
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class BCSQuestionAnalyzer:
    def __init__(self):
        """Initialize the BCS Question Analyzer"""
        self.data = []
        self.subjects = {}
        self.topic_frequency = defaultdict(int)
        self.year_wise_topics = defaultdict(list)
        self.knowledge_graph = nx.Graph()
        self.question_patterns = {}
        self.subject_weights = {}
        
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
            nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            nltk.download('punkt')
            nltk.download('stopwords')
            nltk.download('punkt_tab')
        
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        
    def parse_bcs_file(self, file_content, year):
        """Parse BCS question file content"""
        lines = file_content.strip().split('\n')
        questions = []
        
        for line in lines:
            if not line.strip() or line.startswith('#'):
                continue
                
            parts = line.split('\t')
            if len(parts) >= 7:
                question_data = {
                    'question_id': parts[0],
                    'question': parts[1],
                    'option_a': parts[2],
                    'option_b': parts[3],
                    'option_c': parts[4],
                    'option_d': parts[5],
                    'correct_answer': parts[6],
                    'explanation': parts[7] if len(parts) > 7 else '',
                    'tags': parts[8] if len(parts) > 8 else '',
                    'year': year
                }
                questions.append(question_data)
        
        return questions
    
    def load_data(self, file_contents_dict):
        """Load data from multiple BCS files"""
        print("Loading BCS question data...")
        
        for year, content in file_contents_dict.items():
            questions = self.parse_bcs_file(content, year)
            self.data.extend(questions)
            
        self.df = pd.DataFrame(self.data)
        print(f"Loaded {len(self.data)} questions from {len(file_contents_dict)} years")
        
        # Extract subject information
        self._extract_subjects()
        
    def _extract_subjects(self):
        """Extract subject categories from tags"""
        for idx, row in self.df.iterrows():
            tags = row['tags']
            if 'BCS_ENG' in tags:
                subject = 'English'
            elif 'BCS_BD' in tags:
                subject = 'Bangla'
            elif 'BCS_MTH' in tags:
                subject = 'Mathematics'
            elif 'BCS_GK' in tags:
                subject = 'General Knowledge'
            elif 'BCS_GK_INT' in tags:
                subject = 'International Affairs'
            else:
                subject = 'General'
            
            self.df.at[idx, 'subject'] = subject
            
            if subject not in self.subjects:
                self.subjects[subject] = []
            self.subjects[subject].append(idx)
    
    def analyze_question_patterns(self):
        """Analyze patterns in questions across years"""
        print("Analyzing question patterns...")
        
        # Question type patterns
        question_types = {
            'fill_blank': r'____|\.\.\.',
            'identify': r'^(Identify|Find|Select)',
            'meaning': r'meaning|means',
            'correct': r'^(What is correct|Which.*correct)',
            'antonym': r'antonym|opposite',
            'synonym': r'synonym|similar',
            'grammar': r'grammar|tense|voice',
            'math_problem': r'\d+.*=|\+|\-|\*|/',
        }
        
        for q_type, pattern in question_types.items():
            count = self.df['question'].str.contains(pattern, case=False, na=False).sum()
            self.question_patterns[q_type] = count
            
        return self.question_patterns
    
    def build_knowledge_graph(self):
        """Build knowledge graph from question topics"""
        print("Building knowledge graph...")
        
        # Extract topics from questions
        for idx, row in self.df.iterrows():
            question = row['question'].lower()
            subject = row['subject']
            year = row['year']
            
            # Extract key terms
            tokens = word_tokenize(question)
            tokens = [self.stemmer.stem(token) for token in tokens 
                     if token.isalpha() and token not in self.stop_words and len(token) > 3]
            
            # Add nodes and edges to knowledge graph
            for token in tokens[:5]:  # Limit to top 5 tokens per question
                if not self.knowledge_graph.has_node(token):
                    self.knowledge_graph.add_node(token, 
                                                subject=subject, 
                                                frequency=1,
                                                years=[year])
                else:
                    self.knowledge_graph.nodes[token]['frequency'] += 1
                    if year not in self.knowledge_graph.nodes[token]['years']:
                        self.knowledge_graph.nodes[token]['years'].append(year)
            
            # Connect related tokens
            for i, token1 in enumerate(tokens[:5]):
                for token2 in tokens[i+1:5]:
                    if self.knowledge_graph.has_edge(token1, token2):
                        self.knowledge_graph.edges[token1, token2]['weight'] += 1
                    else:
                        self.knowledge_graph.add_edge(token1, token2, weight=1)
    
    def calculate_subject_importance(self):
        """Calculate importance of each subject based on question frequency"""
        subject_counts = self.df['subject'].value_counts()
        total_questions = len(self.df)
        
        for subject, count in subject_counts.items():
            importance = count / total_questions
            trend = self._calculate_subject_trend(subject)
            difficulty = self._estimate_subject_difficulty(subject)
            
            self.subject_weights[subject] = {
                'frequency': count,
                'importance': importance,
                'trend': trend,
                'difficulty': difficulty,
                'priority_score': importance * (1 + trend) * difficulty
            }
        
        return self.subject_weights
    
    def _calculate_subject_trend(self, subject):
        """Calculate if subject questions are increasing/decreasing over years"""
        subject_data = self.df[self.df['subject'] == subject]
        year_counts = subject_data.groupby('year').size()
        
        if len(year_counts) < 2:
            return 0
        
        years = sorted(year_counts.index)
        recent_avg = year_counts[years[-2:]].mean() if len(years) >= 2 else year_counts.iloc[-1]
        overall_avg = year_counts.mean()
        
        return (recent_avg - overall_avg) / overall_avg if overall_avg > 0 else 0
    
    def _estimate_subject_difficulty(self, subject):
        """Estimate subject difficulty based on question complexity"""
        subject_questions = self.df[self.df['subject'] == subject]['question']
        
        difficulty_indicators = {
            'long_questions': subject_questions.str.len().mean() / 100,
            'complex_words': subject_questions.str.count(r'[A-Z]{2,}').mean(),
            'numbers': subject_questions.str.count(r'\d+').mean()
        }
        
        return min(sum(difficulty_indicators.values()) / len(difficulty_indicators), 2.0)
    
    def predict_next_questions(self, num_predictions=20):
        """Predict probable questions for next exam"""
        print("Generating predictions for next exam...")
        
        predictions = []
        
        # Analyze patterns for each subject
        for subject in self.subjects.keys():
            subject_data = self.df[self.df['subject'] == subject]
            
            # Find common question patterns
            common_patterns = self._find_common_patterns(subject_data)
            
            # Generate predictions based on patterns
            subject_predictions = self._generate_subject_predictions(
                subject, common_patterns, 
                max(3, num_predictions // len(self.subjects))
            )
            predictions.extend(subject_predictions)
        
        return predictions[:num_predictions]
    
    def _find_common_patterns(self, subject_data):
        """Find common patterns in subject questions"""
        questions = subject_data['question'].tolist()
        
        # Use TF-IDF to find important terms
        vectorizer = TfidfVectorizer(max_features=50, stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(questions)
        feature_names = vectorizer.get_feature_names_out()
        
        # Get top terms
        mean_scores = np.mean(tfidf_matrix.toarray(), axis=0)
        top_indices = np.argsort(mean_scores)[-10:]
        top_terms = [feature_names[i] for i in top_indices]
        
        return {
            'top_terms': top_terms,
            'question_types': self._analyze_question_types(questions),
            'common_structures': self._find_common_structures(questions)
        }
    
    def _analyze_question_types(self, questions):
        """Analyze types of questions in the list"""
        types = {
            'fill_blank': 0,
            'multiple_choice': 0,
            'identify': 0,
            'meaning': 0,
            'correct_form': 0
        }
        
        for q in questions:
            q_lower = q.lower()
            if '____' in q or '...' in q:
                types['fill_blank'] += 1
            elif q_lower.startswith(('identify', 'find', 'select')):
                types['identify'] += 1
            elif 'meaning' in q_lower or 'means' in q_lower:
                types['meaning'] += 1
            elif 'correct' in q_lower:
                types['correct_form'] += 1
            else:
                types['multiple_choice'] += 1
        
        return types
    
    def _find_common_structures(self, questions):
        """Find common question structures"""
        structures = []
        
        for q in questions[:10]:  # Analyze first 10 questions
            # Simplify structure
            structure = re.sub(r'"[^"]*"', '"QUOTED"', q)
            structure = re.sub(r'\b\d+\b', 'NUMBER', structure)
            structure = re.sub(r'\b[A-Z][a-z]+\b', 'PROPER_NOUN', structure)
            structures.append(structure)
        
        return Counter(structures).most_common(5)
    
    def _generate_subject_predictions(self, subject, patterns, count):
        """Generate predictions for a specific subject"""
        predictions = []
        
        templates = {
            'English': [
                'What is the correct form of: "{term}"?',
                'Identify the {term} in the sentence.',
                'The meaning of "{term}" is:',
                'Fill in the blank: The {term} _____ yesterday.',
                'Choose the correct {term}:'
            ],
            'Bangla': [
                '"{term}" শব্দের বিপরীত অর্থ কী?',
                'কোনটি {term} এর সঠিক রূপ?',
                '"{term}" এর অর্থ কী?',
                'সঠিক বানান কোনটি?'
            ],
            'Mathematics': [
                'If {term} = NUMBER, then what is the value?',
                'Solve: {term} + NUMBER = ?',
                'The {term} of NUMBER is:',
                'Calculate the {term}:'
            ],
            'General Knowledge': [
                'Who is known as {term}?',
                'When was {term} established?',
                'What is the {term} of Bangladesh?',
                'Which {term} is located in Bangladesh?'
            ]
        }
        
        subject_templates = templates.get(subject, templates['General Knowledge'])
        top_terms = patterns['top_terms']
        
        for i in range(min(count, len(subject_templates))):
            template = subject_templates[i % len(subject_templates)]
            term = top_terms[i % len(top_terms)] if top_terms else 'concept'
            
            question = template.replace('{term}', term)
            predictions.append({
                'subject': subject,
                'predicted_question': question,
                'confidence': 0.7 + (i * 0.05),  # Mock confidence score
                'based_on_pattern': template
            })
        
        return predictions
    
    def generate_study_recommendations(self):
        """Generate study recommendations based on analysis"""
        recommendations = {
            'high_priority_subjects': [],
            'topics_to_focus': [],
            'question_types_to_practice': [],
            'weak_areas': []
        }
        
        # High priority subjects
        sorted_subjects = sorted(self.subject_weights.items(), 
                               key=lambda x: x[1]['priority_score'], 
                               reverse=True)
        
        recommendations['high_priority_subjects'] = [
            {
                'subject': subject,
                'priority_score': data['priority_score'],
                'reason': f"High frequency ({data['frequency']} questions) with "
                         f"{'increasing' if data['trend'] > 0 else 'stable'} trend"
            }
            for subject, data in sorted_subjects[:3]
        ]
        
        # Important topics from knowledge graph
        important_nodes = sorted(self.knowledge_graph.nodes(data=True), 
                               key=lambda x: x[1]['frequency'], 
                               reverse=True)[:10]
        
        recommendations['topics_to_focus'] = [
            {
                'topic': node,
                'frequency': data['frequency'],
                'subjects': data.get('subject', 'Multiple'),
                'recent_years': sorted(data.get('years', []))[-3:]
            }
            for node, data in important_nodes
        ]
        
        # Question types to practice
        sorted_patterns = sorted(self.question_patterns.items(), 
                               key=lambda x: x[1], 
                               reverse=True)
        
        recommendations['question_types_to_practice'] = [
            {
                'type': q_type.replace('_', ' ').title(),
                'frequency': count,
                'importance': 'High' if count > len(self.data) * 0.1 else 'Medium'
            }
            for q_type, count in sorted_patterns[:5]
        ]
        
        return recommendations
    
    def visualize_analysis(self):
        """Create visualizations for the analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Subject distribution
        subject_counts = self.df['subject'].value_counts()
        axes[0, 0].pie(subject_counts.values, labels=subject_counts.index, autopct='%1.1f%%')
        axes[0, 0].set_title('Subject Distribution')
        
        # Year-wise question count
        year_counts = self.df['year'].value_counts().sort_index()
        axes[0, 1].bar(year_counts.index.astype(str), year_counts.values)
        axes[0, 1].set_title('Questions per Year')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Question patterns
        pattern_names = list(self.question_patterns.keys())
        pattern_counts = list(self.question_patterns.values())
        axes[1, 0].barh(pattern_names, pattern_counts)
        axes[1, 0].set_title('Question Type Patterns')
        
        # Subject priority scores
        if self.subject_weights:
            subjects = list(self.subject_weights.keys())
            priority_scores = [data['priority_score'] for data in self.subject_weights.values()]
            axes[1, 1].bar(subjects, priority_scores)
            axes[1, 1].set_title('Subject Priority Scores')
            axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
        
        # Knowledge graph visualization
        if len(self.knowledge_graph.nodes()) > 0:
            plt.figure(figsize=(12, 8))
            pos = nx.spring_layout(self.knowledge_graph, k=1, iterations=50)
            
            # Node sizes based on frequency
            node_sizes = [self.knowledge_graph.nodes[node].get('frequency', 1) * 100 
                         for node in self.knowledge_graph.nodes()]
            
            nx.draw(self.knowledge_graph, pos, 
                   node_size=node_sizes,
                   node_color='lightblue',
                   with_labels=True,
                   font_size=8,
                   font_weight='bold')
            
            plt.title('Knowledge Graph - Topic Relationships')
            plt.axis('off')
            plt.show()
    
    def export_results(self, filename='bcs_analysis_results.txt'):
        """Export analysis results to a file"""
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("BCS Question Analysis Results\n")
            f.write("=" * 50 + "\n\n")
            
            # Summary statistics
            f.write("SUMMARY STATISTICS\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total questions analyzed: {len(self.df)}\n")
            f.write(f"Number of subjects: {len(self.subjects)}\n")
            f.write(f"Years covered: {sorted(self.df['year'].unique())}\n\n")
            
            # Subject weights
            f.write("SUBJECT PRIORITY ANALYSIS\n")
            f.write("-" * 25 + "\n")
            for subject, data in sorted(self.subject_weights.items(), 
                                      key=lambda x: x[1]['priority_score'], 
                                      reverse=True):
                f.write(f"{subject}:\n")
                f.write(f"  Frequency: {data['frequency']} questions\n")
                f.write(f"  Importance: {data['importance']:.3f}\n")
                f.write(f"  Priority Score: {data['priority_score']:.3f}\n\n")
            
            # Predictions
            predictions = self.predict_next_questions()
            f.write("PREDICTED QUESTIONS FOR NEXT EXAM\n")
            f.write("-" * 35 + "\n")
            for i, pred in enumerate(predictions, 1):
                f.write(f"{i}. [{pred['subject']}] {pred['predicted_question']}\n")
                f.write(f"   Confidence: {pred['confidence']:.2f}\n\n")
    
    def run_full_analysis(self, file_contents_dict):
        """Run complete analysis pipeline"""
        print("Starting BCS Question Analysis...")
        
        # Load data
        self.load_data(file_contents_dict)
        
        # Analyze patterns
        self.analyze_question_patterns()
        
        # Build knowledge graph
        self.build_knowledge_graph()
        
        # Calculate subject importance
        self.calculate_subject_importance()
        
        # Generate recommendations
        recommendations = self.generate_study_recommendations()
        
        # Generate predictions
        predictions = self.predict_next_questions()
        
        print("\nAnalysis completed successfully!")
        return {
            'subject_weights': self.subject_weights,
            'predictions': predictions,
            'recommendations': recommendations,
            'question_patterns': self.question_patterns
        }

# Example usage and demonstration
if __name__ == "__main__":
    # Initialize analyzer
    analyzer = BCSQuestionAnalyzer()
    
    # Example data structure for multiple files
    # In practice, you would read these from actual files
    sample_files = {
        42: """1. What is the correct indirect form of: He said, "You had better see a doctor"	He advised him to see a doctor.	He advised that he should see a doctor.	He suggested that he seen a doctor.	He proposed to see a doctor.	He advised him to see a doctor.	প্রশ্নোক্ত direct speech টির subject second person হওয়ায় এবং 'had better' থাকায় indirect speech এর গঠন হবে- Subject + advise/advised + object + infinitive + বাকি অংশ।	BCS_42 BCS_ENG""",
        41: """2. Identify the word that remains the same in plural form.	deer	horse	elephant	tiger	deer	কতগুলো শব্দ আছে যাদের singular এবং plural form একই রকম থাকে।	BCS_41 BCS_ENG""",
        40: """3. Which word is correct?	Furnitures	Informations	Sceneries	Proceeds	Proceeds	Non-count Noun plural হয় না।	BCS_40 BCS_ENG"""
    }
    
    print("BCS Question Analysis System")
    print("=" * 40)
    print("\nDemo with sample data...")
    
    # Run analysis with sample data
    results = analyzer.run_full_analysis(sample_files)
    
    # Display results
    print("\n" + "="*50)
    print("ANALYSIS RESULTS")
    print("="*50)
    
    print("\nSUBJECT PRIORITIES:")
    for subject, data in results['subject_weights'].items():
        print(f"  {subject}: Priority Score = {data['priority_score']:.3f}")
    
    print(f"\nPREDICTED QUESTIONS (Top 5):")
    for i, pred in enumerate(results['predictions'][:5], 1):
        print(f"  {i}. [{pred['subject']}] {pred['predicted_question']}")
    
    print(f"\nSTUDY RECOMMENDATIONS:")
    recs = results['recommendations']
    print("  High Priority Subjects:")
    for rec in recs['high_priority_subjects']:
        print(f"    - {rec['subject']} (Score: {rec['priority_score']:.3f})")
    
    print("\n" + "="*50)
    print("To use with your data:")
    print("1. Prepare file_contents_dict with your BCS files")
    print("2. Call analyzer.run_full_analysis(file_contents_dict)")
    print("3. Use analyzer.visualize_analysis() for charts")
    print("4. Call analyzer.export_results() to save results")
    print("="*50)
