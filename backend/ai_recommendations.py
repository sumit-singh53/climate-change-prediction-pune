"""
AI-Powered Climate Recommendations Module
Generates intelligent recommendations based on climate data and predictions
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import json
import os

class ClimateRecommendationEngine:
    """
    AI-powered recommendation engine for climate adaptation strategies
    """
    
    def __init__(self):
        self.recommendation_templates = {
            'temperature': {
                'high_risk': [
                    "Implement urban heat island mitigation strategies including increased green cover",
                    "Develop early warning systems for extreme heat events",
                    "Promote cool roofing and reflective building materials",
                    "Enhance urban ventilation corridors and wind flow patterns",
                    "Establish public cooling centers during heat waves"
                ],
                'moderate_risk': [
                    "Increase tree plantation in urban areas by {percentage}%",
                    "Promote energy-efficient cooling systems in buildings",
                    "Implement water conservation measures for hot weather",
                    "Develop heat-resilient infrastructure standards"
                ],
                'low_risk': [
                    "Continue monitoring temperature trends",
                    "Maintain existing green infrastructure",
                    "Promote sustainable urban planning practices"
                ]
            },
            'rainfall': {
                'high_variability': [
                    "Develop comprehensive flood management systems",
                    "Implement rainwater harvesting infrastructure",
                    "Create drought-resistant water storage facilities",
                    "Establish early warning systems for extreme weather",
                    "Improve drainage and stormwater management"
                ],
                'moderate_variability': [
                    "Enhance water storage capacity by {percentage}%",
                    "Implement smart irrigation systems",
                    "Develop climate-resilient agriculture practices",
                    "Create buffer zones for flood protection"
                ],
                'low_variability': [
                    "Maintain current water management systems",
                    "Continue monitoring precipitation patterns",
                    "Optimize existing water distribution networks"
                ]
            },
            'air_quality': {
                'poor': [
                    "Implement strict vehicle emission standards",
                    "Promote electric vehicle adoption with {percentage}% incentives",
                    "Establish low emission zones in city centers",
                    "Enhance public transportation infrastructure",
                    "Implement industrial emission control measures"
                ],
                'moderate': [
                    "Increase monitoring of air quality parameters",
                    "Promote cycling and walking infrastructure",
                    "Implement green building standards",
                    "Encourage renewable energy adoption"
                ],
                'good': [
                    "Maintain current air quality standards",
                    "Continue monitoring pollution levels",
                    "Promote sustainable transportation options"
                ]
            },
            'general': [
                "Develop climate-resilient infrastructure standards",
                "Implement comprehensive climate monitoring systems",
                "Create community awareness programs about climate change",
                "Establish emergency response protocols for extreme weather",
                "Promote sustainable development practices"
            ]
        }
        
        self.impact_calculations = {
            'green_cover_temperature': 0.5,  # °C reduction per 10% green cover increase
            'tree_plantation_co2': 22,  # kg CO2 absorbed per tree per year
            'renewable_energy_emissions': 0.85,  # % reduction in emissions
            'public_transport_emissions': 0.45,  # % reduction in transport emissions
            'rainwater_harvesting_water': 0.3,  # % increase in water availability
        }
    
    def analyze_temperature_risk(self, data: pd.DataFrame, predictions: Optional[pd.DataFrame] = None) -> Dict:
        """Analyze temperature-related climate risks"""
        analysis = {
            'current_avg': data['temperature'].mean() if 'temperature' in data.columns else 0,
            'risk_level': 'low',
            'trend': 'stable',
            'recommendations': []
        }
        
        if 'temperature' in data.columns:
            current_avg = data['temperature'].mean()
            
            # Calculate trend if we have yearly data
            if 'year' in data.columns and len(data['year'].unique()) > 1:
                yearly_temp = data.groupby('year')['temperature'].mean()
                trend_slope = np.polyfit(yearly_temp.index, yearly_temp.values, 1)[0]
                
                if trend_slope > 0.1:
                    analysis['trend'] = 'warming'
                elif trend_slope < -0.1:
                    analysis['trend'] = 'cooling'
                
                analysis['trend_rate'] = trend_slope
            
            # Determine risk level
            if current_avg > 32:
                analysis['risk_level'] = 'high_risk'
            elif current_avg > 28:
                analysis['risk_level'] = 'moderate_risk'
            else:
                analysis['risk_level'] = 'low_risk'
            
            # Add predictions analysis
            if predictions is not None and 'temperature_predicted' in predictions.columns:
                future_avg = predictions['temperature_predicted'].mean()
                temp_increase = future_avg - current_avg
                
                if temp_increase > 2:
                    analysis['risk_level'] = 'high_risk'
                elif temp_increase > 1:
                    analysis['risk_level'] = 'moderate_risk'
                
                analysis['predicted_increase'] = temp_increase
        
        return analysis
    
    def analyze_rainfall_variability(self, data: pd.DataFrame, predictions: Optional[pd.DataFrame] = None) -> Dict:
        """Analyze rainfall variability and patterns"""
        analysis = {
            'annual_avg': 0,
            'variability': 'low',
            'seasonal_pattern': {},
            'recommendations': []
        }
        
        if 'rainfall' in data.columns and 'year' in data.columns:
            # Calculate annual rainfall
            annual_rainfall = data.groupby('year')['rainfall'].sum()
            analysis['annual_avg'] = annual_rainfall.mean()
            
            # Calculate coefficient of variation
            cv = annual_rainfall.std() / annual_rainfall.mean()
            
            if cv > 0.4:
                analysis['variability'] = 'high_variability'
            elif cv > 0.2:
                analysis['variability'] = 'moderate_variability'
            else:
                analysis['variability'] = 'low_variability'
            
            analysis['coefficient_variation'] = cv
            
            # Seasonal analysis
            if 'season' in data.columns:
                seasonal_rainfall = data.groupby('season')['rainfall'].sum()
                analysis['seasonal_pattern'] = seasonal_rainfall.to_dict()
            
            # Predictions analysis
            if predictions is not None and 'rainfall_predicted' in predictions.columns:
                future_avg = predictions['rainfall_predicted'].sum()
                current_total = data['rainfall'].sum()
                rainfall_change = ((future_avg - current_total) / current_total) * 100
                
                analysis['predicted_change_percent'] = rainfall_change
                
                if abs(rainfall_change) > 20:
                    analysis['variability'] = 'high_variability'
        
        return analysis
    
    def analyze_air_quality(self, data: pd.DataFrame) -> Dict:
        """Analyze air quality trends and risks"""
        analysis = {
            'avg_aqi': 0,
            'quality_level': 'good',
            'trend': 'stable',
            'recommendations': []
        }
        
        if 'aqi' in data.columns:
            avg_aqi = data['aqi'].mean()
            analysis['avg_aqi'] = avg_aqi
            
            # Determine quality level
            if avg_aqi > 150:
                analysis['quality_level'] = 'poor'
            elif avg_aqi > 100:
                analysis['quality_level'] = 'moderate'
            else:
                analysis['quality_level'] = 'good'
            
            # Calculate trend
            if 'year' in data.columns and len(data['year'].unique()) > 1:
                yearly_aqi = data.groupby('year')['aqi'].mean()
                trend_slope = np.polyfit(yearly_aqi.index, yearly_aqi.values, 1)[0]
                
                if trend_slope > 2:
                    analysis['trend'] = 'worsening'
                elif trend_slope < -2:
                    analysis['trend'] = 'improving'
                
                analysis['trend_rate'] = trend_slope
        
        return analysis
    
    def calculate_intervention_impact(self, intervention: str, scale: float) -> Dict:
        """Calculate potential impact of climate interventions"""
        impacts = {}
        
        if intervention == 'green_cover_increase':
            temp_reduction = scale * self.impact_calculations['green_cover_temperature'] / 10
            co2_absorption = scale * 1000 * self.impact_calculations['tree_plantation_co2']  # Assuming 1000 trees per % increase
            
            impacts = {
                'temperature_reduction_celsius': round(temp_reduction, 2),
                'co2_absorption_kg_per_year': round(co2_absorption, 0),
                'description': f"Increasing green cover by {scale}% could reduce temperature by {temp_reduction:.1f}°C and absorb {co2_absorption:,.0f} kg CO2 annually"
            }
        
        elif intervention == 'renewable_energy':
            emission_reduction = scale * self.impact_calculations['renewable_energy_emissions'] / 100
            
            impacts = {
                'emission_reduction_percent': round(emission_reduction * 100, 1),
                'description': f"Adopting {scale}% renewable energy could reduce emissions by {emission_reduction*100:.1f}%"
            }
        
        elif intervention == 'public_transport':
            transport_emission_reduction = scale * self.impact_calculations['public_transport_emissions'] / 100
            
            impacts = {
                'transport_emission_reduction_percent': round(transport_emission_reduction * 100, 1),
                'description': f"Improving public transport by {scale}% could reduce transport emissions by {transport_emission_reduction*100:.1f}%"
            }
        
        elif intervention == 'rainwater_harvesting':
            water_increase = scale * self.impact_calculations['rainwater_harvesting_water'] / 100
            
            impacts = {
                'water_availability_increase_percent': round(water_increase * 100, 1),
                'description': f"Implementing rainwater harvesting at {scale}% scale could increase water availability by {water_increase*100:.1f}%"
            }
        
        return impacts
    
    def generate_comprehensive_recommendations(self, data: pd.DataFrame, 
                                            predictions: Optional[Dict[str, pd.DataFrame]] = None,
                                            risk_score: Optional[int] = None) -> Dict:
        """Generate comprehensive climate recommendations"""
        print("🧠 Generating AI-powered climate recommendations...")
        
        recommendations = {
            'priority_actions': [],
            'medium_term_strategies': [],
            'long_term_planning': [],
            'intervention_impacts': {},
            'implementation_timeline': {},
            'cost_benefit_analysis': {}
        }
        
        # Analyze different climate aspects
        temp_analysis = self.analyze_temperature_risk(data, 
            predictions.get('temperature') if predictions else None)
        
        rainfall_analysis = self.analyze_rainfall_variability(data,
            predictions.get('rainfall') if predictions else None)
        
        aqi_analysis = self.analyze_air_quality(data)
        
        # Generate priority actions based on highest risks
        priority_actions = []
        
        # Temperature-based recommendations
        if temp_analysis['risk_level'] == 'high_risk':
            priority_actions.extend(self.recommendation_templates['temperature']['high_risk'][:2])
            
            # Calculate specific green cover recommendation
            if 'predicted_increase' in temp_analysis:
                green_cover_needed = min(30, temp_analysis['predicted_increase'] * 10)
                impact = self.calculate_intervention_impact('green_cover_increase', green_cover_needed)
                priority_actions.append(f"Increase urban green cover by {green_cover_needed:.0f}% - {impact['description']}")
                recommendations['intervention_impacts']['green_cover'] = impact
        
        elif temp_analysis['risk_level'] == 'moderate_risk':
            priority_actions.extend(self.recommendation_templates['temperature']['moderate_risk'][:2])
        
        # Air quality recommendations
        if aqi_analysis['quality_level'] == 'poor':
            priority_actions.extend(self.recommendation_templates['air_quality']['poor'][:2])
            
            # Calculate EV adoption impact
            ev_impact = self.calculate_intervention_impact('public_transport', 50)
            priority_actions.append(f"Promote electric vehicle adoption - {ev_impact['description']}")
            recommendations['intervention_impacts']['electric_vehicles'] = ev_impact
        
        # Rainfall variability recommendations
        if rainfall_analysis['variability'] == 'high_variability':
            priority_actions.extend(self.recommendation_templates['rainfall']['high_variability'][:2])
            
            # Calculate rainwater harvesting impact
            rwh_impact = self.calculate_intervention_impact('rainwater_harvesting', 25)
            priority_actions.append(f"Implement citywide rainwater harvesting - {rwh_impact['description']}")
            recommendations['intervention_impacts']['rainwater_harvesting'] = rwh_impact
        
        # Medium-term strategies
        medium_term = []
        medium_term.extend(self.recommendation_templates['general'][:3])
        
        if temp_analysis.get('trend') == 'warming':
            medium_term.append("Develop heat-resilient building codes and standards")
        
        if aqi_analysis['trend'] == 'worsening':
            medium_term.append("Implement comprehensive air quality monitoring network")
        
        # Long-term planning
        long_term = [
            "Develop climate-resilient urban master plan for 2050",
            "Establish climate adaptation fund for infrastructure upgrades",
            "Create regional climate cooperation frameworks",
            "Implement nature-based solutions for climate resilience"
        ]
        
        # Implementation timeline
        timeline = {
            'immediate_0_6_months': priority_actions[:3],
            'short_term_6_18_months': priority_actions[3:] + medium_term[:2],
            'medium_term_1_5_years': medium_term[2:],
            'long_term_5_plus_years': long_term
        }
        
        # Cost-benefit analysis (simplified)
        cost_benefit = {
            'green_cover_increase': {
                'cost_per_hectare': 50000,  # USD
                'benefit_temperature_reduction': 'High',
                'benefit_air_quality': 'Medium',
                'benefit_biodiversity': 'High',
                'roi_years': 5
            },
            'renewable_energy': {
                'cost_per_mw': 1000000,  # USD
                'benefit_emission_reduction': 'High',
                'benefit_energy_security': 'High',
                'roi_years': 8
            },
            'public_transport': {
                'cost_per_km': 20000000,  # USD
                'benefit_emission_reduction': 'Medium',
                'benefit_air_quality': 'Medium',
                'benefit_mobility': 'High',
                'roi_years': 15
            }
        }
        
        recommendations.update({
            'priority_actions': priority_actions,
            'medium_term_strategies': medium_term,
            'long_term_planning': long_term,
            'implementation_timeline': timeline,
            'cost_benefit_analysis': cost_benefit,
            'analysis_summary': {
                'temperature_risk': temp_analysis,
                'rainfall_variability': rainfall_analysis,
                'air_quality': aqi_analysis
            }
        })
        
        print(f"✅ Generated {len(priority_actions)} priority actions and comprehensive strategy")
        return recommendations
    
    def format_recommendations_for_display(self, recommendations: Dict) -> List[str]:
        """Format recommendations for display in dashboard"""
        formatted = []
        
        # Priority actions
        formatted.append("🚨 PRIORITY ACTIONS:")
        for i, action in enumerate(recommendations['priority_actions'][:5], 1):
            formatted.append(f"   {i}. {action}")
        
        # Key interventions with impact
        if recommendations['intervention_impacts']:
            formatted.append("\n💡 KEY INTERVENTIONS & IMPACT:")
            for intervention, impact in recommendations['intervention_impacts'].items():
                formatted.append(f"   • {impact['description']}")
        
        # Timeline
        formatted.append("\n📅 IMPLEMENTATION TIMELINE:")
        timeline = recommendations['implementation_timeline']
        formatted.append(f"   Immediate (0-6 months): {len(timeline.get('immediate_0_6_months', []))} actions")
        formatted.append(f"   Short-term (6-18 months): {len(timeline.get('short_term_6_18_months', []))} actions")
        formatted.append(f"   Medium-term (1-5 years): {len(timeline.get('medium_term_1_5_years', []))} strategies")
        formatted.append(f"   Long-term (5+ years): {len(timeline.get('long_term_5_plus_years', []))} plans")
        
        return formatted


def generate_ai_recommendations(data: pd.DataFrame, 
                              predictions: Optional[Dict[str, pd.DataFrame]] = None,
                              risk_score: Optional[int] = None) -> Dict:
    """
    Main function to generate AI-powered climate recommendations
    
    Args:
        data: Historical climate data
        predictions: Dictionary of prediction DataFrames
        risk_score: Overall climate risk score
    
    Returns:
        Dictionary with comprehensive recommendations
    """
    print("🤖 GENERATING AI-POWERED CLIMATE RECOMMENDATIONS")
    print("=" * 60)
    
    engine = ClimateRecommendationEngine()
    recommendations = engine.generate_comprehensive_recommendations(data, predictions, risk_score)
    
    print(f"\n🎯 RECOMMENDATION SUMMARY:")
    print(f"   🚨 Priority actions: {len(recommendations['priority_actions'])}")
    print(f"   📊 Intervention impacts: {len(recommendations['intervention_impacts'])}")
    print(f"   📅 Implementation phases: {len(recommendations['implementation_timeline'])}")
    
    return recommendations


# Example usage and testing
if __name__ == "__main__":
    # Create sample data for testing
    dates = pd.date_range('2020-01-01', '2024-12-31', freq='D')
    sample_data = {
        'date': dates,
        'temperature': 28 + 3 * np.sin(2 * np.pi * np.arange(len(dates)) / 365) + np.random.normal(0, 2, len(dates)),
        'rainfall': np.random.exponential(2, len(dates)),
        'aqi': 85 + np.random.normal(0, 20, len(dates)),
        'year': [d.year for d in dates],
        'season': ['Winter' if m in [12,1,2] else 'Summer' if m in [3,4,5] else 'Monsoon' if m in [6,7,8,9] else 'Post-Monsoon' for m in [d.month for d in dates]]
    }
    
    df_sample = pd.DataFrame(sample_data)
    
    # Test recommendation generation
    recommendations = generate_ai_recommendations(df_sample, risk_score=65)
    
    print(f"\n✅ AI recommendations test completed!")
    print(f"Generated comprehensive climate strategy with {len(recommendations['priority_actions'])} priority actions")