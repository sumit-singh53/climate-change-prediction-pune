"""
Report Generation Module
Generates downloadable PDF reports including graphs, metrics, and AI-generated insights
"""

import pandas as pd
import numpy as np
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts.linecharts import HorizontalLineChart
from reportlab.graphics.charts.barcharts import VerticalBarChart
from datetime import datetime
import io
import base64
from typing import Dict, List, Any, Optional
import os
import tempfile
import warnings
warnings.filterwarnings('ignore')

class ClimateReportGenerator:
    """Advanced PDF report generator for climate analysis"""
    
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self.custom_styles = self._create_custom_styles()
        
    def _create_custom_styles(self) -> Dict[str, ParagraphStyle]:
        """Create custom paragraph styles"""
        custom_styles = {}
        
        # Title style
        custom_styles['CustomTitle'] = ParagraphStyle(
            'CustomTitle',
            parent=self.styles['Title'],
            fontSize=24,
            spaceAfter=30,
            textColor=colors.darkblue,
            alignment=1  # Center alignment
        )
        
        # Heading style
        custom_styles['CustomHeading'] = ParagraphStyle(
            'CustomHeading',
            parent=self.styles['Heading1'],
            fontSize=16,
            spaceAfter=12,
            textColor=colors.darkgreen,
            borderWidth=1,
            borderColor=colors.darkgreen,
            borderPadding=5
        )
        
        # Subheading style
        custom_styles['CustomSubheading'] = ParagraphStyle(
            'CustomSubheading',
            parent=self.styles['Heading2'],
            fontSize=14,
            spaceAfter=10,
            textColor=colors.darkblue
        )
        
        # Metric style
        custom_styles['MetricStyle'] = ParagraphStyle(
            'MetricStyle',
            parent=self.styles['Normal'],
            fontSize=12,
            spaceAfter=6,
            leftIndent=20
        )
        
        # Insight style
        custom_styles['InsightStyle'] = ParagraphStyle(
            'InsightStyle',
            parent=self.styles['Normal'],
            fontSize=11,
            spaceAfter=8,
            leftIndent=15,
            textColor=colors.darkslategray,
            backColor=colors.lightgrey,
            borderWidth=1,
            borderColor=colors.grey,
            borderPadding=8
        )
        
        return custom_styles
    
    def _create_summary_table(self, data: Dict[str, Any]) -> Table:
        """Create a summary statistics table"""
        table_data = [['Metric', 'Value', 'Description']]
        
        # Add data rows
        for key, value in data.items():
            if isinstance(value, (int, float)):
                if isinstance(value, float):
                    formatted_value = f"{value:.3f}"
                else:
                    formatted_value = str(value)
                
                # Add description based on metric
                descriptions = {
                    'r2': 'Coefficient of determination (0-1, higher is better)',
                    'rmse': 'Root Mean Square Error (lower is better)',
                    'mae': 'Mean Absolute Error (lower is better)',
                    'mape': 'Mean Absolute Percentage Error (%)',
                    'correlation': 'Pearson correlation coefficient',
                    'sample_size': 'Number of data points used',
                    'performance_grade': 'Overall model performance rating'
                }
                
                description = descriptions.get(key, 'Model performance metric')
                table_data.append([key.upper(), formatted_value, description])
        
        # Create table
        table = Table(table_data, colWidths=[2*inch, 1.5*inch, 3*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey])
        ]))
        
        return table
    
    def _generate_insights(self, data: pd.DataFrame, predictions: Dict[str, pd.DataFrame] = None,
                          model_results: Dict[str, Any] = None) -> List[str]:
        """Generate AI-powered insights from the data"""
        insights = []
        
        # Data period insights
        if 'date' in data.columns:
            start_date = data['date'].min()
            end_date = data['date'].max()
            years_span = (end_date - start_date).days / 365.25
            insights.append(f"Analysis covers {years_span:.1f} years of climate data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}.")
        
        # Temperature insights
        if 'temperature' in data.columns:
            temp_mean = data['temperature'].mean()
            temp_trend = data.groupby('year')['temperature'].mean().diff().mean() if 'year' in data.columns else 0
            
            if temp_trend > 0.05:
                insights.append(f"Temperature shows a warming trend of approximately {temp_trend:.2f}°C per year, indicating climate change impact.")
            elif temp_trend < -0.05:
                insights.append(f"Temperature shows a cooling trend of {abs(temp_trend):.2f}°C per year.")
            else:
                insights.append(f"Temperature remains relatively stable with average of {temp_mean:.1f}°C.")
        
        # Rainfall insights
        if 'rainfall' in data.columns:
            rainfall_annual = data.groupby('year')['rainfall'].sum() if 'year' in data.columns else data['rainfall']
            rainfall_cv = rainfall_annual.std() / rainfall_annual.mean()
            
            if rainfall_cv > 0.3:
                insights.append("Rainfall patterns show high variability, indicating potential climate instability.")
            else:
                insights.append("Rainfall patterns are relatively consistent year-over-year.")
        
        # Seasonal insights
        if 'season' in data.columns and 'temperature' in data.columns:
            seasonal_temp = data.groupby('season')['temperature'].mean()
            hottest_season = seasonal_temp.idxmax()
            coldest_season = seasonal_temp.idxmin()
            insights.append(f"Pune experiences hottest temperatures during {hottest_season} ({seasonal_temp[hottest_season]:.1f}°C) and coolest during {coldest_season} ({seasonal_temp[coldest_season]:.1f}°C).")
        
        # Air quality insights
        if 'aqi' in data.columns:
            aqi_mean = data['aqi'].mean()
            if aqi_mean > 100:
                insights.append(f"Air quality is concerning with average AQI of {aqi_mean:.0f}, indicating unhealthy air conditions.")
            elif aqi_mean > 50:
                insights.append(f"Air quality is moderate with average AQI of {aqi_mean:.0f}.")
            else:
                insights.append(f"Air quality is good with average AQI of {aqi_mean:.0f}.")
        
        # Model performance insights
        if model_results:
            for target, results in model_results.items():
                if 'individual_results' in results:
                    best_model = results.get('best_model', 'Unknown')
                    best_r2 = 0
                    
                    for model_name, model_result in results['individual_results'].items():
                        if model_name == best_model and 'metrics' in model_result:
                            best_r2 = model_result['metrics'].get('r2', 0)
                            break
                    
                    insights.append(f"For {target} prediction, {best_model} performs best with R² score of {best_r2:.3f}, explaining {best_r2*100:.1f}% of variance.")
        
        # Future predictions insights
        if predictions:
            for var, pred_df in predictions.items():
                if not pred_df.empty and f'{var}_predicted' in pred_df.columns:
                    future_mean = pred_df[f'{var}_predicted'].mean()
                    current_mean = data[var].tail(365).mean() if var in data.columns else 0
                    
                    if var == 'temperature':
                        change = future_mean - current_mean
                        if change > 1:
                            insights.append(f"Temperature is projected to increase by {change:.1f}°C on average, indicating significant warming.")
                        elif change < -1:
                            insights.append(f"Temperature is projected to decrease by {abs(change):.1f}°C on average.")
                    
                    elif var == 'rainfall':
                        change_pct = ((future_mean - current_mean) / current_mean) * 100 if current_mean > 0 else 0
                        if abs(change_pct) > 10:
                            direction = "increase" if change_pct > 0 else "decrease"
                            insights.append(f"Rainfall patterns are projected to {direction} by {abs(change_pct):.1f}% compared to recent years.")
        
        # Climate risk assessment
        risk_factors = []
        if 'temperature' in data.columns and data['temperature'].mean() > 30:
            risk_factors.append("high temperatures")
        if 'aqi' in data.columns and data['aqi'].mean() > 100:
            risk_factors.append("poor air quality")
        if 'rainfall' in data.columns:
            rainfall_cv = data.groupby('year')['rainfall'].sum().std() / data.groupby('year')['rainfall'].sum().mean() if 'year' in data.columns else 0
            if rainfall_cv > 0.3:
                risk_factors.append("irregular rainfall patterns")
        
        if risk_factors:
            insights.append(f"Climate risk factors identified: {', '.join(risk_factors)}. Adaptation strategies recommended.")
        
        return insights
    
    def _create_recommendations(self, insights: List[str], data: pd.DataFrame) -> List[str]:
        """Generate actionable recommendations based on insights"""
        recommendations = []
        
        # Temperature-based recommendations
        if 'temperature' in data.columns:
            temp_mean = data['temperature'].mean()
            if temp_mean > 32:
                recommendations.append("Implement heat mitigation strategies: increase green cover, promote cool roofing, and enhance urban ventilation.")
                recommendations.append("Develop early warning systems for extreme heat events to protect vulnerable populations.")
        
        # Air quality recommendations
        if 'aqi' in data.columns:
            aqi_mean = data['aqi'].mean()
            if aqi_mean > 100:
                recommendations.append("Strengthen air pollution control measures: promote electric vehicles, improve industrial emission standards.")
                recommendations.append("Enhance public transportation and encourage cycling/walking infrastructure.")
        
        # Rainfall recommendations
        if 'rainfall' in data.columns and 'year' in data.columns:
            rainfall_cv = data.groupby('year')['rainfall'].sum().std() / data.groupby('year')['rainfall'].sum().mean()
            if rainfall_cv > 0.3:
                recommendations.append("Improve water management: build rainwater harvesting systems and enhance drainage infrastructure.")
                recommendations.append("Develop drought and flood preparedness plans for extreme weather events.")
        
        # General climate adaptation
        recommendations.append("Invest in climate-resilient infrastructure and building codes adapted to changing conditions.")
        recommendations.append("Promote renewable energy adoption to reduce local carbon footprint and heat island effects.")
        recommendations.append("Establish continuous climate monitoring systems for better prediction and response.")
        
        return recommendations
    
    def generate_comprehensive_report(self, data: pd.DataFrame, 
                                    predictions: Dict[str, pd.DataFrame] = None,
                                    model_results: Dict[str, Any] = None,
                                    visuals: Dict[str, Any] = None,
                                    output_path: str = None) -> str:
        """Generate comprehensive climate analysis report"""
        print("📄 Generating comprehensive climate report...")
        
        if output_path is None:
            output_path = f"climate_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        
        # Create PDF document
        doc = SimpleDocTemplate(output_path, pagesize=A4, topMargin=1*inch)
        story = []
        
        # Title page
        story.append(Paragraph("Climate Change Analysis Report", self.custom_styles['CustomTitle']))
        story.append(Paragraph("Pune, India", self.custom_styles['CustomTitle']))
        story.append(Spacer(1, 0.5*inch))
        
        # Report metadata
        report_date = datetime.now().strftime("%B %d, %Y")
        story.append(Paragraph(f"Report Generated: {report_date}", self.styles['Normal']))
        
        if 'date' in data.columns:
            data_period = f"{data['date'].min().strftime('%Y-%m-%d')} to {data['date'].max().strftime('%Y-%m-%d')}"
            story.append(Paragraph(f"Data Period: {data_period}", self.styles['Normal']))
            story.append(Paragraph(f"Total Records: {len(data):,}", self.styles['Normal']))
        
        story.append(PageBreak())
        
        # Executive Summary
        story.append(Paragraph("Executive Summary", self.custom_styles['CustomHeading']))
        
        # Generate insights
        insights = self._generate_insights(data, predictions, model_results)
        
        for insight in insights[:5]:  # Top 5 insights for executive summary
            story.append(Paragraph(f"• {insight}", self.custom_styles['InsightStyle']))
            story.append(Spacer(1, 6))
        
        story.append(PageBreak())
        
        # Data Overview
        story.append(Paragraph("Data Overview", self.custom_styles['CustomHeading']))
        
        # Basic statistics table
        if len(data.select_dtypes(include=[np.number]).columns) > 0:
            numeric_cols = ['temperature', 'rainfall', 'humidity', 'aqi']
            available_cols = [col for col in numeric_cols if col in data.columns]
            
            if available_cols:
                stats_data = data[available_cols].describe().round(2)
                
                # Create statistics table
                table_data = [['Variable'] + list(stats_data.index)]
                for col in available_cols:
                    row = [col.title()] + [str(stats_data.loc[stat, col]) for stat in stats_data.index]
                    table_data.append(row)
                
                stats_table = Table(table_data)
                stats_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 10),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black),
                    ('FONTSIZE', (0, 1), (-1, -1), 8)
                ]))
                
                story.append(stats_table)
                story.append(Spacer(1, 20))
        
        # Model Performance (if available)
        if model_results:
            story.append(Paragraph("Model Performance Analysis", self.custom_styles['CustomHeading']))
            
            for target, results in model_results.items():
                story.append(Paragraph(f"{target.title()} Prediction Models", self.custom_styles['CustomSubheading']))
                
                if 'individual_results' in results:
                    for model_name, model_result in results['individual_results'].items():
                        if 'metrics' in model_result:
                            metrics = model_result['metrics']
                            story.append(Paragraph(f"<b>{model_name}:</b>", self.styles['Normal']))
                            story.append(Paragraph(f"R² Score: {metrics.get('r2', 'N/A'):.3f}", self.custom_styles['MetricStyle']))
                            story.append(Paragraph(f"RMSE: {metrics.get('rmse', 'N/A'):.3f}", self.custom_styles['MetricStyle']))
                            story.append(Paragraph(f"MAE: {metrics.get('mae', 'N/A'):.3f}", self.custom_styles['MetricStyle']))
                            story.append(Spacer(1, 10))
                
                if 'best_model' in results:
                    story.append(Paragraph(f"<b>Best Model:</b> {results['best_model']}", self.styles['Normal']))
                    story.append(Spacer(1, 15))
        
        story.append(PageBreak())
        
        # Detailed Insights
        story.append(Paragraph("Detailed Climate Analysis", self.custom_styles['CustomHeading']))
        
        for insight in insights:
            story.append(Paragraph(f"• {insight}", self.custom_styles['InsightStyle']))
            story.append(Spacer(1, 8))
        
        story.append(PageBreak())
        
        # Recommendations
        story.append(Paragraph("Recommendations", self.custom_styles['CustomHeading']))
        
        recommendations = self._create_recommendations(insights, data)
        
        for i, recommendation in enumerate(recommendations, 1):
            story.append(Paragraph(f"{i}. {recommendation}", self.styles['Normal']))
            story.append(Spacer(1, 10))
        
        story.append(PageBreak())
        
        # Climate Risk Index
        story.append(Paragraph("Climate Risk Assessment", self.custom_styles['CustomHeading']))
        
        # Calculate risk index
        risk_score = 0
        risk_factors = []
        
        if 'temperature' in data.columns:
            temp_mean = data['temperature'].mean()
            if temp_mean > 35:
                risk_score += 3
                risk_factors.append("Extreme heat risk")
            elif temp_mean > 30:
                risk_score += 2
                risk_factors.append("High temperature risk")
        
        if 'aqi' in data.columns:
            aqi_mean = data['aqi'].mean()
            if aqi_mean > 150:
                risk_score += 3
                risk_factors.append("Severe air pollution")
            elif aqi_mean > 100:
                risk_score += 2
                risk_factors.append("Unhealthy air quality")
        
        if 'rainfall' in data.columns and 'year' in data.columns:
            rainfall_cv = data.groupby('year')['rainfall'].sum().std() / data.groupby('year')['rainfall'].sum().mean()
            if rainfall_cv > 0.4:
                risk_score += 2
                risk_factors.append("Irregular precipitation patterns")
        
        # Risk level classification
        if risk_score >= 6:
            risk_level = "HIGH"
            risk_color = colors.red
        elif risk_score >= 3:
            risk_level = "MEDIUM"
            risk_color = colors.orange
        else:
            risk_level = "LOW"
            risk_color = colors.green
        
        story.append(Paragraph(f"<b>Climate Risk Level: <font color='{risk_color}'>{risk_level}</font></b>", self.styles['Normal']))
        story.append(Paragraph(f"Risk Score: {risk_score}/9", self.styles['Normal']))
        story.append(Spacer(1, 15))
        
        story.append(Paragraph("Risk Factors Identified:", self.styles['Normal']))
        for factor in risk_factors:
            story.append(Paragraph(f"• {factor}", self.custom_styles['MetricStyle']))
        
        if not risk_factors:
            story.append(Paragraph("• No significant climate risks identified", self.custom_styles['MetricStyle']))
        
        # Footer
        story.append(PageBreak())
        story.append(Paragraph("Report Disclaimer", self.custom_styles['CustomHeading']))
        disclaimer = """This report is generated based on available climate data and predictive models. 
        Predictions are estimates and should be used in conjunction with other sources for decision-making. 
        Climate patterns are subject to natural variability and global climate change influences."""
        story.append(Paragraph(disclaimer, self.styles['Normal']))
        
        # Build PDF
        doc.build(story)
        
        print(f"✅ Report generated: {output_path}")
        return output_path


def generate_report(data: pd.DataFrame, predictions: Dict[str, pd.DataFrame] = None,
                   model_results: Dict[str, Any] = None, visuals: Dict[str, Any] = None,
                   output_path: str = None) -> str:
    """
    Main function to generate comprehensive climate report
    
    Args:
        data: Historical climate data
        predictions: Dictionary of prediction DataFrames
        model_results: Dictionary of model evaluation results
        visuals: Dictionary of visualization figures
        output_path: Output file path for the PDF report
    
    Returns:
        Path to the generated PDF report
    """
    print("📊 GENERATING COMPREHENSIVE CLIMATE REPORT")
    print("=" * 60)
    
    generator = ClimateReportGenerator()
    
    report_path = generator.generate_comprehensive_report(
        data=data,
        predictions=predictions,
        model_results=model_results,
        visuals=visuals,
        output_path=output_path
    )
    
    print(f"\n📄 REPORT SUMMARY:")
    print(f"   📁 File: {report_path}")
    print(f"   📊 Data records: {len(data):,}")
    
    if predictions:
        print(f"   🔮 Predictions: {list(predictions.keys())}")
    
    if model_results:
        total_models = sum(len(results.get('individual_results', {})) for results in model_results.values())
        print(f"   🤖 Models evaluated: {total_models}")
    
    return report_path


# Example usage and testing
if __name__ == "__main__":
    # Create sample data for testing
    dates = pd.date_range('2020-01-01', '2024-12-31', freq='D')
    sample_data = {
        'date': dates,
        'temperature': 25 + 5 * np.sin(2 * np.pi * np.arange(len(dates)) / 365) + np.random.normal(0, 2, len(dates)),
        'rainfall': np.random.exponential(2, len(dates)),
        'humidity': 60 + 20 * np.sin(2 * np.pi * np.arange(len(dates)) / 365) + np.random.normal(0, 5, len(dates)),
        'aqi': 70 + np.random.normal(0, 15, len(dates)),
        'year': [d.year for d in dates],
        'month': [d.month for d in dates],
        'season': ['Winter' if m in [12,1,2] else 'Summer' if m in [3,4,5] else 'Monsoon' if m in [6,7,8,9] else 'Post-Monsoon' for m in [d.month for d in dates]]
    }
    
    df_sample = pd.DataFrame(sample_data)
    
    # Test report generation
    report_path = generate_report(df_sample, output_path="test_climate_report.pdf")
    
    print(f"\n✅ Report generation test completed!")
    print(f"Test report saved as: {report_path}")