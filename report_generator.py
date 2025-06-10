import io
import json
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging
import traceback

# Import reportlab components at a higher scope
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors


class EnhancedReportGenerator:
    """Enhanced PDF report generator with comprehensive error handling"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        # Configure matplotlib for PDF generation
        plt.style.use('default')
        sns.set_palette("husl")
        # Set figure parameters
        plt.rcParams['figure.max_open_warning'] = 0
        plt.rcParams['font.size'] = 10
        
        # Initialize ReportLab styles
        self.styles = getSampleStyleSheet()
        self._add_custom_styles(self.styles)

    def _add_custom_styles(self, styles):
        """Adds custom paragraph styles for the report."""
        styles.add(ParagraphStyle(
            name='Heading1Centered',
            parent=styles['h1'],
            alignment=1, # TA_CENTER
            spaceAfter=14
        ))
        styles.add(ParagraphStyle(
            name='Heading2Centered',
            parent=styles['h2'],
            alignment=1, # TA_CENTER
            spaceAfter=12
        ))
        styles.add(ParagraphStyle(
            name='NormalCenter',
            parent=styles['Normal'],
            alignment=1, # TA_CENTER
            spaceAfter=6
        ))
        styles.add(ParagraphStyle(
            name='BulletPoint',
            parent=styles['Normal'],
            leftIndent=0.5 * inch,
            firstLineIndent=-0.25 * inch,
            spaceBefore=6,
            bulletText='•'
        ))
        styles.add(ParagraphStyle(
            name='Subheading',
            parent=styles['h3'],
            fontSize=12,
            spaceBefore=12,
            spaceAfter=6
        ))
        styles.add(ParagraphStyle(
            name='CodeBlock',
            parent=styles['Code'],
            fontSize=8,
            leading=9,
            spaceBefore=6,
            spaceAfter=6,
            borderWidth=0.5,
            borderColor=colors.lightgrey,
            backColor=colors.lavender,
            fontName='Courier', # Use a monospaced font
            alignment=0 # TA_LEFT
        ))


    def _validate_inputs(self, recommendations: Dict) -> bool:
        """Validate the recommendations input structure."""
        if not isinstance(recommendations, dict) or not recommendations:
            self.logger.error("Validation Error: Recommendations must be a non-empty dictionary.")
            return False
        for env, rec in recommendations.items():
            if not isinstance(rec, dict):
                self.logger.error(f"Validation Error: Recommendation for '{env}' is not a dictionary.")
                return False
            # Check for either single instance structure or new Multi-AZ structure
            if 'instance_type' not in rec and 'writer' not in rec:
                self.logger.error(f"Validation Error: Recommendation for '{env}' missing 'instance_type' or 'writer' key.")
                return False
            if 'total_cost' not in rec:
                self.logger.error(f"Validation Error: Recommendation for '{env}' missing 'total_cost' key.")
                return False
        return True

    def generate_enhanced_pdf_report(self, recommendations: Dict, target_engine: str, ai_insights: Dict = None) -> bytes:
        """Generate PDF with comprehensive error handling and validation - Enhanced interface"""
        
        try:
            # Validate inputs first
            if not self._validate_inputs(recommendations):
                return self._generate_error_pdf("Invalid input data provided for PDF generation.")
            
            buffer = io.BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=A4)
            elements = []
            
            # --- Title Page ---
            elements.append(Paragraph("AWS RDS Migration & Sizing Report", self.styles['Heading1Centered']))
            elements.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", self.styles['NormalCenter']))
            elements.append(Spacer(1, 0.5 * inch))
            elements.append(Paragraph("Comprehensive Analysis & Recommendations", self.styles['Heading2Centered']))
            elements.append(Spacer(1, 1 * inch))
            
            # Add a placeholder for a logo or image (optional)
            # try:
            #     img_path = "path/to/your/logo.png" # Replace with actual logo path if available
            #     img = Image(img_path)
            #     img.drawHeight = 1 * inch
            #     img.drawWidth = 4 * inch
            #     img.hAlign = 'CENTER'
            #     elements.append(img)
            #     elements.append(Spacer(1, 0.5 * inch))
            # except FileNotFoundError:
            #     self.logger.warning(f"Logo image not found at {img_path}")
            
            elements.append(Paragraph(f"Target Database Engine: {target_engine.upper()}", self.styles['NormalCenter']))
            elements.append(Spacer(1, 2 * inch))
            elements.append(Paragraph("Prepared for Enterprise Cloud Solutions", self.styles['NormalCenter']))
            elements.append(PageBreak())

            # --- Executive Summary ---
            elements.append(Paragraph("Executive Summary", self.styles['Heading1']))
            elements.append(Spacer(1, 0.2 * inch))
            
            prod_rec = recommendations.get("PROD")
            if prod_rec and 'error' not in prod_rec:
                total_monthly_cost = self._get_total_cost(prod_rec)
                elements.append(Paragraph(f"This report provides a comprehensive analysis for migrating your on-premises database environment to AWS RDS, focusing on the target engine {target_engine.upper()}.", self.styles['Normal']))
                elements.append(Paragraph(f"Our analysis indicates an estimated total monthly cost for the production environment of **${total_monthly_cost:,.2f}**, leading to an annual cost of **${total_monthly_cost*12:,.2f}**.", self.styles['Normal']))
                
                # Dynamic instance type for summary
                if 'writer' in prod_rec: # Multi-AZ / Cluster setup
                    writer_type = self._get_instance_type(prod_rec['writer'])
                    reader_types = [self._get_instance_type(r) for r in prod_rec.get('readers', []) if isinstance(r, dict)]
                    reader_summary = f" and {len(reader_types)} reader(s) ({', '.join(set(reader_types))})" if reader_types else ""
                    elements.append(Paragraph(f"The recommended architecture for production includes a primary writer instance of type **{writer_type}**{reader_summary}.", self.styles['Normal']))
                    elements.append(Paragraph(f"Deployment Option: **{prod_rec.get('deployment_option', 'Multi-AZ')}** for high availability and scalability.", self.styles['Normal']))
                else: # Single instance setup
                    instance_type = self._get_instance_type(prod_rec)
                    elements.append(Paragraph(f"The recommended production instance type is **{instance_type}**.", self.styles['Normal']))
                    elements.append(Paragraph(f"Deployment Option: **{prod_rec.get('deployment_option', 'Single-AZ')}**.", self.styles['Normal']))

            elements.append(Spacer(1, 0.2 * inch))
            if ai_insights and "ai_analysis" in ai_insights:
                elements.append(Paragraph("AI-Powered Insights:", self.styles['Subheading']))
                elements.append(Paragraph(ai_insights["ai_analysis"], self.styles['Normal']))
            
            elements.append(PageBreak())

            # --- Detailed Recommendations ---
            elements.append(Paragraph("Detailed Recommendations by Environment", self.styles['Heading1']))
            elements.append(Spacer(1, 0.2 * inch))

            for env, rec in recommendations.items():
                elements.append(Paragraph(f"{env} Environment Recommendations", self.styles['Heading2']))
                elements.append(Spacer(1, 0.1 * inch))

                if 'error' in rec:
                    elements.append(Paragraph(f"Error calculating recommendations for {env}: {rec['error']}", self.styles['Normal']))
                    elements.append(Spacer(1, 0.2 * inch))
                    continue
                
                # Check for Multi-AZ structure
                if 'writer' in rec:
                    elements.append(Paragraph(f"Deployment Option: <b>{rec.get('deployment_option', 'Multi-AZ')}</b>", self.styles['Normal']))
                    elements.append(Spacer(1, 0.1 * inch))

                    # Writer Instance Details
                    elements.append(Paragraph("Primary Writer Instance:", self.styles['Subheading']))
                    writer_data = [
                        ['Metric', 'Recommended', 'Actual'],
                        ['Instance Type', self._get_instance_type(rec['writer']), self._get_instance_type(rec['writer'])],
                        ['vCPUs', self._format_value(rec['writer'].get('vCPUs')), self._format_value(rec['writer'].get('actual_vCPUs'))],
                        ['RAM (GB)', self._format_value(rec['writer'].get('RAM_GB')), self._format_value(rec['writer'].get('actual_RAM_GB'))]
                    ]
                    writer_table_style = TableStyle([
                        ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
                        ('TEXTCOLOR', (0,0), (-1,0), colors.black),
                        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
                        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                        ('BOTTOMPADDING', (0,0), (-1,0), 8),
                        ('BACKGROUND', (0,1), (-1,-1), colors.beige),
                        ('GRID', (0,0), (-1,-1), 0.5, colors.grey)
                    ])
                    elements.append(Table(writer_data, style=writer_table_style))
                    elements.append(Spacer(1, 0.1 * inch))

                    # Reader Instances Details
                    if rec.get('readers'):
                        elements.append(Paragraph("Reader Instances:", self.styles['Subheading']))
                        for i, reader in enumerate(rec['readers']):
                            elements.append(Paragraph(f"Reader {i+1}:", self.styles['Normal']))
                            reader_data = [
                                ['Metric', 'Recommended', 'Actual'],
                                ['Instance Type', self._get_instance_type(reader), self._get_instance_type(reader)],
                                ['vCPUs', self._format_value(reader.get('vCPUs')), self._format_value(reader.get('actual_vCPUs'))],
                                ['RAM (GB)', self._format_value(reader.get('RAM_GB')), self._format_value(reader.get('actual_RAM_GB'))]
                            ]
                            reader_table_style = TableStyle([
                                ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
                                ('TEXTCOLOR', (0,0), (-1,0), colors.black),
                                ('ALIGN', (0,0), (-1,-1), 'LEFT'),
                                ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                                ('BOTTOMPADDING', (0,0), (-1,0), 8),
                                ('BACKGROUND', (0,1), (-1,-1), colors.lightgreen if i % 2 == 0 else colors.lightcyan),
                                ('GRID', (0,0), (-1,-1), 0.5, colors.grey)
                            ])
                            elements.append(Table(reader_data, style=reader_table_style))
                            elements.append(Spacer(1, 0.1 * inch))
                else: # Single Instance structure
                    elements.append(Paragraph(f"Deployment Option: <b>{rec.get('deployment_option', 'Single-AZ')}</b>", self.styles['Normal']))
                    elements.append(Spacer(1, 0.1 * inch))
                    elements.append(Paragraph("Recommended Instance:", self.styles['Subheading']))
                    instance_data = [
                        ['Metric', 'Recommended', 'Actual'],
                        ['Instance Type', self._get_instance_type(rec), self._get_instance_type(rec)],
                        ['vCPUs', self._format_value(rec.get('vCPUs')), self._format_value(rec.get('actual_vCPUs'))],
                        ['RAM (GB)', self._format_value(rec.get('RAM_GB')), self._format_value(rec.get('actual_RAM_GB'))]
                    ]
                    instance_table_style = TableStyle([
                        ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
                        ('TEXTCOLOR', (0,0), (-1,0), colors.black),
                        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
                        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                        ('BOTTOMPADDING', (0,0), (-1,0), 8),
                        ('BACKGROUND', (0,1), (-1,-1), colors.aliceblue),
                        ('GRID', (0,0), (-1,-1), 0.5, colors.grey)
                    ])
                    elements.append(Table(instance_data, style=instance_table_style))
                    elements.append(Spacer(1, 0.1 * inch))

                # Common details for all deployment types
                elements.append(Paragraph("Storage & Backup:", self.styles['Subheading']))
                elements.append(Paragraph(f"Storage (GB): {self._format_value(rec.get('storage_GB'))}", self.styles['Normal']))
                elements.append(Spacer(1, 0.1 * inch))

                elements.append(Paragraph("Cost Summary (Monthly):", self.styles['Subheading']))
                cost_breakdown = rec.get('cost_breakdown', {})
                if 'writer_monthly' in cost_breakdown: # Multi-AZ structure
                    elements.append(Paragraph(f"Writer Instance Cost: ${cost_breakdown.get('writer_monthly', 0):,.2f}", self.styles['Normal']))
                    elements.append(Paragraph(f"Reader Instances Cost: ${cost_breakdown.get('readers_monthly', 0):,.2f}", self.styles['Normal']))
                else: # Single instance structure
                    elements.append(Paragraph(f"Instance Cost: ${rec.get('instance_cost', 0):,.2f}", self.styles['Normal']))
                
                elements.append(Paragraph(f"Storage Cost: ${rec.get('storage_cost', 0):,.2f}", self.styles['Normal']))
                elements.append(Paragraph(f"Backup Cost: ${rec.get('backup_cost', 0):,.2f}", self.styles['Normal']))
                elements.append(Paragraph(f"Migration Related Costs: ${safe_get(cost_breakdown, 'migration_monthly', 0):,.2f}", self.styles['Normal']))
                elements.append(Paragraph(f"Features & Data Transfer Costs: ${safe_get(cost_breakdown, 'features_monthly', 0) + safe_get(cost_breakdown, 'data_transfer_monthly', 0):,.2f}", self.styles['Normal']))
                
                elements.append(Paragraph(f"<b>Total Monthly Cost: ${rec.get('total_cost', 0):,.2f}</b>", self.styles['Normal']))
                elements.append(Paragraph(f"<b>Total Annual Cost: ${rec.get('total_cost', 0)*12:,.2f}</b>", self.styles['Normal']))
                elements.append(Spacer(1, 0.2 * inch))

                if rec.get('advisories'):
                    elements.append(Paragraph("Advisories:", self.styles['Subheading']))
                    for advisory in rec['advisories']:
                        elements.append(Paragraph(advisory, self.styles['BulletPoint']))
                    elements.append(Spacer(1, 0.2 * inch))
                
                elements.append(PageBreak())

            # --- AI Insights and Overall Summary ---
            if ai_insights:
                elements.append(Paragraph("AI-Powered Insights Summary", self.styles['Heading1']))
                elements.append(Spacer(1, 0.2 * inch))

                if "ai_analysis" in ai_insights:
                    elements.append(Paragraph("Overall AI Analysis:", self.styles['Subheading']))
                    elements.append(Paragraph(ai_insights["ai_analysis"], self.styles['Normal']))
                    elements.append(Spacer(1, 0.2 * inch))
                
                elements.append(Paragraph("Key AI Metrics:", self.styles['Subheading']))
                elements.append(Paragraph(f"Migration Risk Level: <b>{ai_insights.get('risk_level', 'N/A')}</b>", self.styles['Normal']))
                elements.append(Paragraph(f"Estimated Cost Optimization Potential: <b>{(ai_insights.get('cost_optimization_potential', 0)*100):.1f}%</b>", self.styles['Normal']))
                
                if 'recommended_writers' in ai_insights and 'recommended_readers' in ai_insights:
                     elements.append(Paragraph(f"AI Recommended Architecture: <b>{ai_insights['recommended_writers']} Writer(s) and {ai_insights['recommended_readers']} Reader(s)</b>", self.styles['Normal']))

                elements.append(Spacer(1, 0.2 * inch))
                
                if "recommended_migration_phases" in ai_insights:
                    elements.append(Paragraph("Recommended Migration Phases:", self.styles['Subheading']))
                    for phase in ai_insights["recommended_migration_phases"]:
                        elements.append(Paragraph(phase, self.styles['BulletPoint']))
                elements.append(PageBreak())
            
            # --- Conclusion ---
            elements.append(Paragraph("Conclusion", self.styles['Heading1']))
            elements.append(Spacer(1, 0.2 * inch))
            elements.append(Paragraph("This report provides a detailed sizing and migration strategy to help you plan your transition to AWS RDS. By leveraging the recommended instance types, storage configurations, and following the migration advisories, you can ensure a performant, cost-effective, and highly available database environment in the cloud.", self.styles['Normal']))
            elements.append(Spacer(1, 0.5 * inch))
            elements.append(Paragraph("Thank you for using our Enterprise AWS RDS Migration & Sizing Tool.", self.styles['NormalCenter']))

            doc.build(elements)
            buffer.seek(0)
            return buffer.getvalue()
            
        except Exception as e:
            error_msg = f"Unexpected error in PDF generation: {str(e)}\n{traceback.format_exc()}"
            self.logger.error(error_msg)
            return self._generate_error_pdf(f"An unexpected error occurred during PDF generation. Please check the logs. Error: {str(e)}")

    def _generate_error_pdf(self, message: str) -> bytes:
        """Generates a simple error PDF."""
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4)
        elements = []
        styles = getSampleStyleSheet()

        elements.append(Paragraph("PDF Generation Error", styles['h1']))
        elements.append(Spacer(1, 0.5 * inch))
        elements.append(Paragraph("An error occurred while generating the PDF report.", styles['Normal']))
        elements.append(Spacer(1, 0.2 * inch))
        elements.append(Paragraph(f"Details: {message}", styles['CodeBlock']))
        
        doc.build(elements)
        buffer.seek(0)
        return buffer.getvalue()

    def _get_total_cost(self, recommendation: Dict) -> float:
        """Helper to get total cost, handling new structure."""
        return safe_get(recommendation, 'total_cost', 0)

    def _get_instance_type(self, recommendation_part: Dict) -> str:
        """Helper to get instance type from a recommendation part (single, writer, or reader)."""
        return safe_get_str(recommendation_part, 'instance_type', 'N/A')

    def _format_value(self, value: Any) -> str:
        """Formats a value for display, handling None or 0 gracefully."""
        if value is None or value == 0:
            return "N/A"
        if isinstance(value, (int, float)):
            return f"{value:,.0f}" if value == int(value) else f"{value:,.2f}"
        return str(value)

# Backwards compatibility aliases
RobustReportGenerator = EnhancedReportGenerator

# Usage function
def generate_pdf_report(calculator, recommendations, target_engine, ai_insights=None):
    """Main function to generate PDF report"""
    generator = EnhancedReportGenerator()
    return generator.generate_enhanced_pdf_report(recommendations, target_engine, ai_insights)

# Helper function for safe dictionary access (copied from Streamlit app to ensure independence)
def safe_get(dictionary, key, default=0):
    """Safely get a value from a dictionary with a default fallback"""
    if isinstance(dictionary, dict):
        return dictionary.get(key, default)
    return default

def safe_get_str(dictionary, key, default="N/A"):
    """Safely get a string value from a dictionary with a default fallback"""
    if isinstance(dictionary, dict):
        return dictionary.get(key, default)
    return default
