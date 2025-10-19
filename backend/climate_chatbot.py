"""
Climate Chatbot Module
Simple AI-powered chatbot for climate-related questions and answers
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import re
import json
from datetime import datetime
import os

class ClimateKnowledgeBase:
    """Knowledge base for climate-related information"""
    
    def __init__(self):
        self.climate_facts = {
            'pune_climate': {
                'temperature': "Pune has a tropical wet and dry climate with temperatures ranging from 15°C to 42°C throughout the year.",
                'seasons': "Pune experiences three main seasons: Winter (Dec-Feb), Summer (Mar-May), and Monsoon (Jun-Nov).",
                'rainfall': "Pune receives an average annual rainfall of 700-800mm, mostly during the monsoon season.",
                'air_quality': "Pune's air quality varies from moderate to poor, with AQI typically ranging from 50-150."
            },
            'climate_change': {
                'global_warming': "Global warming refers to the long-term increase in Earth's average surface temperature due to greenhouse gas emissions.",
                'greenhouse_gases': "Main greenhouse gases include CO2, methane, nitrous oxide, and fluorinated gases.",
                'impacts': "Climate change impacts include rising temperatures, changing precipitation patterns, extreme weather events, and sea level rise.",
                'mitigation': "Climate mitigation involves reducing greenhouse gas emissions through renewable energy, energy efficiency, and sustainable practices."
            },
            'adaptation': {
                'urban_planning': "Climate-resilient urban planning includes green infrastructure, sustainable drainage, and heat island mitigation.",
                'water_management': "Adaptive water management includes rainwater harvesting, efficient irrigation, and drought-resistant systems.",
                'energy': "Climate adaptation in energy involves renewable sources, smart grids, and energy-efficient buildings.",
                'agriculture': "Climate-smart agriculture includes drought-resistant crops, efficient irrigation, and soil conservation."
            }
        }
        
        self.response_patterns = {
            'temperature': [
                r'temperature|temp|hot|cold|heat|warm|cool',
                r'how hot|how warm|how cold',
                r'degrees|celsius|fahrenheit'
            ],
            'rainfall': [
                r'rain|rainfall|precipitation|monsoon|wet|dry',
                r'how much rain|rainfall amount',
                r'mm|millimeter|inches'
            ],
            'air_quality': [
                r'air quality|aqi|pollution|pollutant|smog',
                r'pm2\.5|pm10|ozone|co2|carbon',
                r'clean air|dirty air|breathe'
            ],
            'climate_change': [
                r'climate change|global warming|greenhouse|carbon footprint',
                r'rising temperature|sea level|extreme weather',
                r'sustainability|renewable|fossil fuel'
            ],
            'predictions': [
                r'future|predict|forecast|projection|trend',
                r'what will happen|how will|expected',
                r'2030|2040|2050|next year|coming years'
            ],
            'recommendations': [
                r'what should|how to|recommend|suggest|advice',
                r'reduce|mitigate|adapt|prevent|improve',
                r'action|solution|strategy|plan'
            ]
        }
        
        self.sample_responses = {
            'greeting': [
                "Hello! I'm your climate assistant. I can help you understand Pune's climate patterns, predictions, and adaptation strategies.",
                "Hi there! Ask me anything about Pune's climate, weather patterns, or climate change impacts.",
                "Welcome! I'm here to help with climate-related questions about Pune and environmental insights."
            ],
            'temperature': [
                "Pune's temperature varies seasonally. Summers can reach 40°C+ while winters drop to 15°C. Climate change is causing gradual warming trends.",
                "Based on our data, Pune's average temperature is around 25-27°C, but it's been increasing due to urban heat island effects and climate change.",
                "Temperature patterns in Pune show warming trends. Summer temperatures are becoming more extreme, requiring adaptation strategies."
            ],
            'rainfall': [
                "Pune receives most rainfall during monsoon (June-September). Annual rainfall averages 700-800mm but shows high variability.",
                "Monsoon patterns are changing in Pune. We're seeing more intense rainfall events and longer dry spells between rains.",
                "Rainfall in Pune is becoming more unpredictable due to climate change, affecting water management and agriculture."
            ],
            'air_quality': [
                "Pune's air quality is a concern with AQI often in moderate to poor range (50-150). Vehicle emissions and construction dust are major contributors.",
                "Air pollution in Pune varies by season and location. Monsoon helps clean the air, while winter months often see higher pollution levels.",
                "Improving air quality in Pune requires reducing vehicle emissions, controlling dust, and increasing green cover."
            ],
            'climate_change': [
                "Climate change is affecting Pune through rising temperatures, changing rainfall patterns, and more extreme weather events.",
                "Pune is experiencing urban heat island effects, irregular monsoons, and air quality challenges due to climate change.",
                "Climate change impacts in Pune include water stress, heat waves, and increased energy demand for cooling."
            ],
            'predictions': [
                "Our models predict Pune may see 2-3°C temperature increase by 2050, with more variable rainfall patterns.",
                "Future projections suggest Pune will face more heat waves, irregular monsoons, and increased water stress.",
                "Climate predictions for Pune indicate need for adaptation in water management, urban planning, and energy systems."
            ],
            'recommendations': [
                "Key recommendations for Pune include increasing green cover, improving public transport, rainwater harvesting, and energy efficiency.",
                "Climate adaptation strategies for Pune: urban forestry, sustainable drainage, renewable energy, and heat-resilient infrastructure.",
                "To address climate challenges, Pune needs integrated planning: green buildings, smart water management, and clean transportation."
            ],
            'default': [
                "That's an interesting question about climate. Could you be more specific about what aspect you'd like to know?",
                "I can help with climate data, predictions, and recommendations for Pune. What specific information are you looking for?",
                "I'm focused on climate and environmental topics for Pune. Could you rephrase your question to be more climate-specific?"
            ]
        }

class ClimateChatbot:
    """Simple rule-based chatbot for climate questions"""
    
    def __init__(self, data: Optional[pd.DataFrame] = None):
        self.knowledge_base = ClimateKnowledgeBase()
        self.data = data
        self.conversation_history = []
        
    def classify_intent(self, message: str) -> str:
        """Classify user intent based on message content"""
        message_lower = message.lower()
        
        # Check for greetings
        if any(word in message_lower for word in ['hello', 'hi', 'hey', 'start', 'help']):
            return 'greeting'
        
        # Check for specific topics
        for topic, patterns in self.knowledge_base.response_patterns.items():
            for pattern in patterns:
                if re.search(pattern, message_lower):
                    return topic
        
        return 'default'
    
    def get_data_insights(self, topic: str) -> Optional[str]:
        """Get insights from actual data if available"""
        if self.data is None or self.data.empty:
            return None
        
        try:
            if topic == 'temperature' and 'temperature' in self.data.columns:
                avg_temp = self.data['temperature'].mean()
                max_temp = self.data['temperature'].max()
                min_temp = self.data['temperature'].min()
                
                return f"Based on our data: Average temperature is {avg_temp:.1f}°C, ranging from {min_temp:.1f}°C to {max_temp:.1f}°C."
            
            elif topic == 'rainfall' and 'rainfall' in self.data.columns:
                total_rainfall = self.data['rainfall'].sum()
                avg_daily = self.data['rainfall'].mean()
                
                return f"Based on our data: Total rainfall recorded is {total_rainfall:.0f}mm with daily average of {avg_daily:.1f}mm."
            
            elif topic == 'air_quality' and 'aqi' in self.data.columns:
                avg_aqi = self.data['aqi'].mean()
                quality_level = "Good" if avg_aqi < 50 else "Moderate" if avg_aqi < 100 else "Poor"
                
                return f"Based on our data: Average AQI is {avg_aqi:.0f} ({quality_level} air quality)."
            
            elif topic == 'predictions' and 'year' in self.data.columns:
                years_span = self.data['year'].max() - self.data['year'].min()
                
                if 'temperature' in self.data.columns and years_span > 1:
                    yearly_temp = self.data.groupby('year')['temperature'].mean()
                    trend = np.polyfit(yearly_temp.index, yearly_temp.values, 1)[0]
                    
                    if trend > 0.1:
                        return f"Based on our data: Temperature is rising at approximately {trend:.2f}°C per year."
                    elif trend < -0.1:
                        return f"Based on our data: Temperature is cooling at approximately {abs(trend):.2f}°C per year."
                    else:
                        return "Based on our data: Temperature trends are relatively stable."
        
        except Exception as e:
            print(f"⚠️ Error getting data insights: {e}")
            return None
        
        return None
    
    def generate_response(self, message: str) -> str:
        """Generate response to user message"""
        intent = self.classify_intent(message)
        
        # Get data-based insights if available
        data_insight = self.get_data_insights(intent)
        
        # Get template response
        if intent in self.knowledge_base.sample_responses:
            template_response = np.random.choice(self.knowledge_base.sample_responses[intent])
        else:
            template_response = np.random.choice(self.knowledge_base.sample_responses['default'])
        
        # Combine data insight with template response
        if data_insight:
            response = f"{data_insight}\n\n{template_response}"
        else:
            response = template_response
        
        # Store conversation
        self.conversation_history.append({
            'timestamp': datetime.now(),
            'user_message': message,
            'bot_response': response,
            'intent': intent
        })
        
        return response
    
    def get_climate_summary(self) -> str:
        """Generate a climate summary based on available data"""
        if self.data is None or self.data.empty:
            return "I don't have specific data to analyze, but I can provide general climate information about Pune."
        
        summary_parts = []
        
        # Temperature summary
        if 'temperature' in self.data.columns:
            avg_temp = self.data['temperature'].mean()
            summary_parts.append(f"🌡️ Average temperature: {avg_temp:.1f}°C")
        
        # Rainfall summary
        if 'rainfall' in self.data.columns:
            total_rainfall = self.data['rainfall'].sum()
            summary_parts.append(f"🌧️ Total rainfall: {total_rainfall:.0f}mm")
        
        # Air quality summary
        if 'aqi' in self.data.columns:
            avg_aqi = self.data['aqi'].mean()
            quality = "Good" if avg_aqi < 50 else "Moderate" if avg_aqi < 100 else "Poor"
            summary_parts.append(f"💨 Air quality: {quality} (AQI: {avg_aqi:.0f})")
        
        # Data period
        if 'date' in self.data.columns:
            start_date = self.data['date'].min().strftime('%Y-%m-%d')
            end_date = self.data['date'].max().strftime('%Y-%m-%d')
            summary_parts.append(f"📅 Data period: {start_date} to {end_date}")
        
        if summary_parts:
            return "📊 **Climate Summary for Pune:**\n" + "\n".join(summary_parts)
        else:
            return "I can provide general climate information about Pune. What would you like to know?"
    
    def get_conversation_history(self) -> List[Dict]:
        """Get conversation history"""
        return self.conversation_history
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []


def create_climate_chatbot(data: Optional[pd.DataFrame] = None) -> ClimateChatbot:
    """
    Create a climate chatbot instance
    
    Args:
        data: Optional climate data for context-aware responses
    
    Returns:
        ClimateChatbot instance
    """
    return ClimateChatbot(data)


def chat_with_bot(chatbot: ClimateChatbot, message: str) -> str:
    """
    Chat with the climate bot
    
    Args:
        chatbot: ClimateChatbot instance
        message: User message
    
    Returns:
        Bot response
    """
    return chatbot.generate_response(message)


# Example usage and testing
if __name__ == "__main__":
    # Create sample data for testing
    dates = pd.date_range('2020-01-01', '2024-12-31', freq='D')
    sample_data = {
        'date': dates,
        'temperature': 25 + 5 * np.sin(2 * np.pi * np.arange(len(dates)) / 365) + np.random.normal(0, 2, len(dates)),
        'rainfall': np.random.exponential(2, len(dates)),
        'aqi': 70 + np.random.normal(0, 15, len(dates)),
        'year': [d.year for d in dates]
    }
    
    df_sample = pd.DataFrame(sample_data)
    
    # Test chatbot
    print("🤖 Testing Climate Chatbot...")
    
    chatbot = create_climate_chatbot(df_sample)
    
    # Test different types of questions
    test_questions = [
        "Hello, what can you tell me about Pune's climate?",
        "What's the average temperature in Pune?",
        "How much rainfall does Pune get?",
        "What about air quality in Pune?",
        "What are the climate predictions for the future?",
        "What should we do about climate change in Pune?"
    ]
    
    print("\n💬 CHATBOT CONVERSATION TEST:")
    print("=" * 50)
    
    for question in test_questions:
        response = chat_with_bot(chatbot, question)
        print(f"\n👤 User: {question}")
        print(f"🤖 Bot: {response}")
        print("-" * 30)
    
    # Test climate summary
    print(f"\n📊 CLIMATE SUMMARY:")
    print(chatbot.get_climate_summary())
    
    print(f"\n✅ Chatbot test completed!")
    print(f"Conversation history: {len(chatbot.get_conversation_history())} exchanges")