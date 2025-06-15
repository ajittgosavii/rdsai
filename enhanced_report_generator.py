import io
import base64
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch, cm
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.charts.piecharts import Pie
from reportlab.graphics.charts.linecharts import HorizontalLineChart
from reportlab.lib.colors import Color

class SuperEnhancedReportGenerator:
    """
    Super Enhanced PDF Report Generator for AWS RDS Migration Tool
    Includes comprehensive graphs, charts, and detailed analysis
    """
    
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self.setup_custom_styles()
        plt.style.use('seaborn-v0_8')  # Set matplotlib style
        
    def setup_custom_styles(self):
        """Setup custom styles for the report"""
        # Title style
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Title'],
            fontSize=28,
            spaceAfter=30,
            textColor=colors.HexColor('#1f4e79'),
            alignment=1,
            fontName='Helvetica-Bold'
        ))
        
        # Section header style
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading1'],
            fontSize=18,
            spaceBefore=25,
            spaceAfter=15,
            textColor=colors.HexColor('#2e75b6'),
            borderWidth=3,
            borderColor=colors.HexColor('#70ad47'),
            borderPadding=8,
            fontName='Helvetica-Bold'
        ))
        
        # Subsection header style
        self.styles.add(ParagraphStyle(
            name='SubsectionHeader',
            parent=self.styles['Heading2'],
            fontSize=14,
            spaceBefore=15,
            spaceAfter=10,
            textColor=colors.HexColor('#70ad47'),
            fontName='Helvetica-Bold'
        ))
        
        # Highlight style
        self.styles.add(ParagraphStyle(
            name='Highlight',
            parent=self.styles['Normal'],
            fontSize=11,
            textColor=colors.HexColor('#c5504b'),
            backColor=colors.HexColor('#fff2cc'),
            borderWidth=1,
            borderColor=colors.HexColor('#d6b656'),
            borderPadding=8,
            leftIndent=20,
            rightIndent=20
        ))
        
        # Key insights style
        self.styles.add(ParagraphStyle(
            name='KeyInsight',
            parent=self.styles['Normal'],
            fontSize=12,
            textColor=colors.HexColor('#1f4e79'),
            backColor=colors.HexColor('#deebf7'),
            borderWidth=2,
            borderColor=colors.HexColor('#2e75b6'),
            borderPadding=10,
            leftIndent=15,
            rightIndent=15,
            spaceBefore=10,
            spaceAfter=10
        ))
        
        # Cost box style
        self.styles.add(ParagraphStyle(
            name='CostBox',
            parent=self.styles['Normal'],
            fontSize=11,
            textColor=colors.HexColor('#ffffff'),
            backColor=colors.HexColor('#70ad47'),
            borderWidth=2,
            borderColor=colors.HexColor('#548235'),
            borderPadding=8,
            alignment=1,
            fontName='Helvetica-Bold'
        ))

    def create_matplotlib_chart(self, chart_type, data, title, labels=None, figsize=(10, 6)):
        """Create a matplotlib chart and return as image"""
        fig, ax = plt.subplots(figsize=figsize)
        
        if chart_type == 'bar':
            bars = ax.bar(data.keys(), data.values(), color=['#2e75b6', '#70ad47', '#c5504b', '#ffc000'])
            ax.set_title(title, fontsize=14, fontweight='bold', color='#1f4e79')
            ax.set_ylabel('Cost ($)', fontweight='bold')
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                       f'${height:,.0f}', ha='center', va='bottom', fontweight='bold')
                       
        elif chart_type == 'pie':
            colors_pie = ['#2e75b6', '#70ad47', '#c5504b', '#ffc000', '#9dc3e6']
            wedges, texts, autotexts = ax.pie(data.values(), labels=data.keys(), autopct='%1.1f%%',
                                            colors=colors_pie, startangle=90)
            ax.set_title(title, fontsize=14, fontweight='bold', color='#1f4e79')
            
            # Enhance text
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
                
        elif chart_type == 'line':
            ax.plot(list(data.keys()), list(data.values()), marker='o', linewidth=3, 
                   markersize=8, color='#2e75b6')
            ax.set_title(title, fontsize=14, fontweight='bold', color='#1f4e79')
            ax.set_ylabel('Cost ($)', fontweight='bold')
            ax.grid(True, alpha=0.3)
            
        elif chart_type == 'stacked_bar':
            # For cost breakdown charts
            bottom = np.zeros(len(list(data.keys())))
            colors_stack = ['#2e75b6', '#70ad47', '#c5504b', '#ffc000']
            
            for i, (category, values) in enumerate(zip(['Instance', 'Storage', 'Backup', 'Transfer'], 
                                                      zip(*data.values()))):
                ax.bar(list(data.keys()), values, bottom=bottom, label=category, 
                      color=colors_stack[i % len(colors_stack)])
                bottom += np.array(values)
                
            ax.set_title(title, fontsize=14, fontweight='bold', color='#1f4e79')
            ax.set_ylabel('Monthly Cost ($)', fontweight='bold')
            ax.legend()
        
        # Common formatting
        ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        
        # Save to BytesIO
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='PNG', dpi=300, bbox_inches='tight')
        img_buffer.seek(0)
        plt.close()
        
        return img_buffer

    def create_cost_comparison_chart(self, analysis_results, analysis_mode):
        """Create comprehensive cost comparison charts"""
        charts = []
        
        if analysis_mode == 'single':
            valid_results = {k: v for k, v in analysis_results.items() if 'error' not in v}
            
            # Environment cost comparison
            env_costs = {}
            cost_breakdown_data = {}
            
            for env, result in valid_results.items():
                total_cost = result.get('total_cost', 0)
                env_costs[env] = total_cost
                
                # Get cost breakdown
                if 'writer' in result:
                    instance_cost = result.get('cost_breakdown', {}).get('writer_monthly', 0) + \
                                  result.get('cost_breakdown', {}).get('readers_monthly', 0)
                    storage_cost = result.get('cost_breakdown', {}).get('storage_monthly', 0)
                    backup_cost = result.get('cost_breakdown', {}).get('backup_monthly', 0)
                else:
                    instance_cost = result.get('instance_cost', 0)
                    storage_cost = result.get('storage_cost', 0)
                    backup_cost = storage_cost * 0.25
                
                cost_breakdown_data[env] = [instance_cost, storage_cost, backup_cost, 0]
            
            # Create charts
            if env_costs:
                chart1 = self.create_matplotlib_chart('bar', env_costs, 
                                                    'Monthly Cost by Environment', figsize=(8, 5))
                charts.append(('env_costs', chart1))
                
                chart2 = self.create_matplotlib_chart('stacked_bar', cost_breakdown_data,
                                                    'Cost Breakdown by Environment', figsize=(8, 5))
                charts.append(('cost_breakdown', chart2))
        
        else:  # bulk analysis
            server_costs = {}
            server_types = {}
            
            for server_name, server_results in analysis_results.items():
                if 'error' not in server_results:
                    result = server_results.get('PROD', list(server_results.values())[0])
                    if 'error' not in result:
                        monthly_cost = result.get('total_cost', 0)
                        server_costs[server_name[:15]] = monthly_cost  # Truncate name
                        
                        if 'writer' in result:
                            instance_type = result['writer'].get('instance_type', 'Unknown')
                        else:
                            instance_type = result.get('instance_type', 'Unknown')
                        
                        server_types[instance_type] = server_types.get(instance_type, 0) + 1
            
            if server_costs:
                chart1 = self.create_matplotlib_chart('bar', server_costs,
                                                    'Monthly Cost by Server', figsize=(12, 6))
                charts.append(('server_costs', chart1))
                
                chart2 = self.create_matplotlib_chart('pie', server_types,
                                                    'Instance Type Distribution', figsize=(8, 8))
                charts.append(('instance_types', chart2))
        
        return charts

    def create_migration_timeline_chart(self):
        """Create migration timeline visualization"""
        phases = ['Assessment', 'Schema Conv.', 'DMS Setup', 'Testing', 'Cutover', 'Optimization']
        durations = [3, 4, 2, 5, 1, 3]  # weeks
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Create Gantt-like chart
        start_date = 0
        colors_timeline = ['#2e75b6', '#70ad47', '#c5504b', '#ffc000', '#9dc3e6', '#bf8f00']
        
        for i, (phase, duration) in enumerate(zip(phases, durations)):
            ax.barh(i, duration, left=start_date, height=0.6, 
                   color=colors_timeline[i], alpha=0.8, edgecolor='black')
            
            # Add phase labels
            ax.text(start_date + duration/2, i, f'{phase}\n({duration}w)', 
                   ha='center', va='center', fontweight='bold', fontsize=10)
            
            start_date += duration
        
        ax.set_yticks(range(len(phases)))
        ax.set_yticklabels(phases)
        ax.set_xlabel('Timeline (Weeks)', fontweight='bold')
        ax.set_title('Migration Timeline & Phases', fontsize=14, fontweight='bold', color='#1f4e79')
        ax.grid(True, axis='x', alpha=0.3)
        
        plt.tight_layout()
        
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='PNG', dpi=300, bbox_inches='tight')
        img_buffer.seek(0)
        plt.close()
        
        return img_buffer

    def create_risk_assessment_chart(self):
        """Create risk assessment matrix"""
        risks = ['Schema\nConversion', 'Performance\nIssues', 'Data\nCorruption', 
                'Extended\nDowntime', 'Cost\nOverrun']
        probability = [2, 1, 1, 2, 2]  # 1=Low, 2=Medium, 3=High
        impact = [3, 2, 4, 3, 2]  # 1=Low, 2=Medium, 3=High, 4=Critical
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create scatter plot
        colors_risk = []
        for p, i in zip(probability, impact):
            risk_score = p * i
            if risk_score <= 2:
                colors_risk.append('#70ad47')  # Low risk - green
            elif risk_score <= 4:
                colors_risk.append('#ffc000')  # Medium risk - yellow
            else:
                colors_risk.append('#c5504b')  # High risk - red
        
        scatter = ax.scatter(probability, impact, s=500, c=colors_risk, alpha=0.7, edgecolors='black')
        
        # Add labels
        for i, risk in enumerate(risks):
            ax.annotate(risk, (probability[i], impact[i]), ha='center', va='center', 
                       fontweight='bold', fontsize=9)
        
        ax.set_xlabel('Probability', fontweight='bold', fontsize=12)
        ax.set_ylabel('Impact', fontweight='bold', fontsize=12)
        ax.set_title('Risk Assessment Matrix', fontsize=14, fontweight='bold', color='#1f4e79')
        
        ax.set_xlim(0.5, 3.5)
        ax.set_ylim(0.5, 4.5)
        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels(['Low', 'Medium', 'High'])
        ax.set_yticks([1, 2, 3, 4])
        ax.set_yticklabels(['Low', 'Medium', 'High', 'Critical'])
        
        ax.grid(True, alpha=0.3)
        
        # Add legend
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#70ad47', 
                                    markersize=10, label='Low Risk'),
                         plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#ffc000', 
                                    markersize=10, label='Medium Risk'),
                         plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#c5504b', 
                                    markersize=10, label='High Risk')]
        ax.legend(handles=legend_elements, loc='upper left')
        
        plt.tight_layout()
        
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='PNG', dpi=300, bbox_inches='tight')
        img_buffer.seek(0)
        plt.close()
        
        return img_buffer

    def create_tco_analysis_chart(self, total_monthly_cost):
        """Create TCO analysis over 5 years"""
        years = list(range(1, 6))
        
        # Calculate costs with 3% annual inflation
        aws_costs = [total_monthly_cost * 12 * (1.03 ** (year - 1)) for year in years]
        
        # Estimated OpEx savings (example values)
        base_savings = 200000 if total_monthly_cost > 5000 else 150000
        opex_savings = [base_savings + (year * 30000) for year in years]
        
        # Net position (negative means savings)
        net_position = [aws - opex for aws, opex in zip(aws_costs, opex_savings)]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Cost comparison chart
        x = np.arange(len(years))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, aws_costs, width, label='AWS Costs', color='#c5504b', alpha=0.8)
        bars2 = ax1.bar(x + width/2, opex_savings, width, label='OpEx Savings', color='#70ad47', alpha=0.8)
        
        ax1.set_xlabel('Year', fontweight='bold')
        ax1.set_ylabel('Cost ($)', fontweight='bold')
        ax1.set_title('5-Year Cost Analysis', fontweight='bold', color='#1f4e79')
        ax1.set_xticks(x)
        ax1.set_xticklabels([f'Year {y}' for y in years])
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                        f'${height:,.0f}', ha='center', va='bottom', fontsize=8, rotation=90)
        
        # Net position chart
        colors_net = ['#70ad47' if x < 0 else '#c5504b' for x in net_position]
        bars3 = ax2.bar(years, net_position, color=colors_net, alpha=0.8)
        ax2.set_xlabel('Year', fontweight='bold')
        ax2.set_ylabel('Net Position ($)', fontweight='bold')
        ax2.set_title('Net Cost Position (AWS - Savings)', fontweight='bold', color='#1f4e79')
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax2.grid(True, alpha=0.3)
        
        # Add value labels
        for bar in bars3:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + (abs(height)*0.02 if height >= 0 else -abs(height)*0.02),
                    f'${height:,.0f}', ha='center', va='bottom' if height >= 0 else 'top', fontsize=9)
        
        plt.tight_layout()
        
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='PNG', dpi=300, bbox_inches='tight')
        img_buffer.seek(0)
        plt.close()
        
        return img_buffer

    def generate_comprehensive_pdf_report(self, analysis_results, analysis_mode, server_specs=None, 
                                        ai_insights=None, transfer_results=None):
        """Generate the super enhanced PDF report"""
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4, 
                              rightMargin=0.5*inch, leftMargin=0.5*inch,
                              topMargin=0.8*inch, bottomMargin=0.8*inch)
        
        story = []
        
        try:
            # 1. ENHANCED TITLE PAGE
            story.extend(self._create_enhanced_title_page(analysis_mode, analysis_results))
            story.append(PageBreak())
            
            # 2. TABLE OF CONTENTS
            story.extend(self._create_table_of_contents())
            story.append(PageBreak())
            
            # 3. EXECUTIVE SUMMARY WITH CHARTS
            story.extend(self._create_enhanced_executive_summary(analysis_results, analysis_mode, ai_insights))
            story.append(PageBreak())
            
            # 4. COST ANALYSIS WITH VISUALIZATIONS
            story.extend(self._create_enhanced_cost_analysis(analysis_results, analysis_mode))
            story.append(PageBreak())
            
            # 5. TECHNICAL SPECIFICATIONS
            story.extend(self._create_enhanced_technical_specs(analysis_results, analysis_mode, server_specs))
            story.append(PageBreak())
            
            # 6. MIGRATION STRATEGY WITH TIMELINE
            story.extend(self._create_enhanced_migration_strategy())
            story.append(PageBreak())
            
            # 7. RISK ASSESSMENT WITH MATRIX
            story.extend(self._create_enhanced_risk_assessment())
            story.append(PageBreak())
            
            # 8. DATA TRANSFER ANALYSIS
            if transfer_results:
                story.extend(self._create_enhanced_transfer_analysis(transfer_results))
                story.append(PageBreak())
            
            # 9. AI INSIGHTS (if available)
            if ai_insights:
                story.extend(self._create_enhanced_ai_insights(ai_insights))
                story.append(PageBreak())
            
            # 10. IMPLEMENTATION ROADMAP
            story.extend(self._create_implementation_roadmap())
            story.append(PageBreak())
            
            # 11. APPENDICES
            story.extend(self._create_appendices(analysis_results, analysis_mode))
            
            # Build PDF
            doc.build(story)
            buffer.seek(0)
            return buffer.getvalue()
            
        except Exception as e:
            print(f"Error generating enhanced PDF: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _create_enhanced_title_page(self, analysis_mode, analysis_results):
        """Create an enhanced title page with key metrics"""
        story = []
        
        # Main title with styling
        title_text = "AWS RDS Migration & Sizing Analysis"
        story.append(Paragraph(title_text, self.styles['CustomTitle']))
        story.append(Spacer(1, 0.3*inch))
        
        # Subtitle
        subtitle = f"Comprehensive {analysis_mode.title()} Server Migration Report"
        story.append(Paragraph(subtitle, self.styles['Heading1']))
        story.append(Spacer(1, 0.5*inch))
        
        # Key metrics summary table
        if analysis_mode == 'single':
            valid_results = {k: v for k, v in analysis_results.items() if 'error' not in v}
            if valid_results:
                prod_result = valid_results.get('PROD', list(valid_results.values())[0])
                total_cost = prod_result.get('total_cost', 0)
                
                metrics_data = [
                    ['Metric', 'Value', 'Impact'],
                    ['Analysis Type', 'Single Server', 'Focused migration approach'],
                    ['Monthly Cost', f'${total_cost:,.2f}', 'Operational expense'],
                    ['Annual Cost', f'${total_cost * 12:,.2f}', 'Total yearly investment'],
                    ['Migration Complexity', 'Medium-High', 'Requires careful planning'],
                    ['Estimated Timeline', '16-24 weeks', 'Including all phases']
                ]
        else:
            total_servers = len(analysis_results)
            successful = sum(1 for r in analysis_results.values() if 'error' not in r)
            total_monthly = sum(r.get('PROD', {}).get('total_cost', 0) for r in analysis_results.values() 
                              if 'error' not in r and 'PROD' in r)
            
            metrics_data = [
                ['Metric', 'Value', 'Impact'],
                ['Analysis Type', 'Bulk Migration', 'Enterprise-scale approach'],
                ['Total Servers', f'{successful}/{total_servers}', 'Migration scope'],
                ['Total Monthly Cost', f'${total_monthly:,.2f}', 'Aggregate operational cost'],
                ['Total Annual Cost', f'${total_monthly * 12:,.2f}', 'Enterprise investment'],
                ['Estimated Timeline', '20-28 weeks', 'Phased migration approach']
            ]
        
        metrics_table = Table(metrics_data, colWidths=[2*inch, 1.8*inch, 2.4*inch])
        metrics_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f4e79')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#deebf7')),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 10)
        ]))
        story.append(metrics_table)
        story.append(Spacer(1, 0.5*inch))
        
        # Report details
        generation_time = datetime.now().strftime("%B %d, %Y at %I:%M %p")
        story.append(Paragraph(f"<b>Generated:</b> {generation_time}", self.styles['Normal']))
        story.append(Paragraph(f"<b>Report Version:</b> Enhanced v2.0 with Visualizations", self.styles['Normal']))
        story.append(Paragraph(f"<b>Prepared for:</b> Enterprise Cloud Migration Team", self.styles['Normal']))
        
        return story

    def _create_table_of_contents(self):
        """Create a table of contents"""
        story = []
        story.append(Paragraph("Table of Contents", self.styles['SectionHeader']))
        
        toc_data = [
            ['Section', 'Page'],
            ['Executive Summary', '3'],
            ['Cost Analysis & Visualizations', '4'],
            ['Technical Specifications', '5'],
            ['Migration Strategy & Timeline', '6'],
            ['Risk Assessment Matrix', '7'],
            ['Data Transfer Analysis', '8'],
            ['AI-Powered Insights', '9'],
            ['Implementation Roadmap', '10'],
            ['Appendices', '11']
        ]
        
        toc_table = Table(toc_data, colWidths=[4*inch, 1*inch])
        toc_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2e75b6')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f2f2f2')),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(toc_table)
        
        return story

    def _create_enhanced_executive_summary(self, analysis_results, analysis_mode, ai_insights):
        """Create enhanced executive summary with key insights"""
        story = []
        story.append(Paragraph("Executive Summary", self.styles['SectionHeader']))
        
        # Key findings box
        if analysis_mode == 'single':
            valid_results = {k: v for k, v in analysis_results.items() if 'error' not in v}
            if valid_results:
                prod_result = valid_results.get('PROD', list(valid_results.values())[0])
                total_cost = prod_result.get('total_cost', 0)
                
                key_findings = f"""
                <b>KEY FINDINGS & RECOMMENDATIONS</b><br/><br/>
                • <b>Migration Feasibility:</b> High - Recommended to proceed with structured approach<br/>
                • <b>Cost Impact:</b> ${total_cost:,.2f}/month operational cost represents competitive positioning<br/>
                • <b>Timeline:</b> 16-24 weeks for complete migration including testing and optimization<br/>
                • <b>Risk Level:</b> {ai_insights.get('risk_level', 'Medium') if ai_insights else 'Medium'} - Manageable with proper planning<br/>
                • <b>ROI Expectation:</b> Positive within 18 months including operational savings
                """
        else:
            total_servers = len(analysis_results)
            successful = sum(1 for r in analysis_results.values() if 'error' not in r)
            total_monthly = sum(r.get('PROD', {}).get('total_cost', 0) for r in analysis_results.values() 
                              if 'error' not in r and 'PROD' in r)
            
            key_findings = f"""
                <b>KEY FINDINGS & RECOMMENDATIONS</b><br/><br/>
                • <b>Migration Scope:</b> {successful} of {total_servers} servers analyzed successfully<br/>
                • <b>Total Investment:</b> ${total_monthly * 12:,.2f} annually for complete infrastructure<br/>
                • <b>Complexity Assessment:</b> Mixed - Requires phased approach with specialized teams<br/>
                • <b>Risk Level:</b> {ai_insights.get('risk_level', 'Medium') if ai_insights else 'Medium'} - Enterprise governance required<br/>
                • <b>Business Impact:</b> Significant modernization with improved scalability and performance
                """
        
        story.append(Paragraph(key_findings, self.styles['KeyInsight']))
        story.append(Spacer(1, 0.3*inch))
        
        # Strategic recommendations
        strategy_text = """
        <b>STRATEGIC RECOMMENDATIONS</b><br/><br/>
        1. <b>Phased Migration Approach:</b> Implement in stages starting with non-critical systems<br/>
        2. <b>Team Preparation:</b> Invest in training and AWS certification for technical teams<br/>
        3. <b>Testing Strategy:</b> Extensive parallel testing to ensure zero-downtime migration<br/>
        4. <b>Cost Optimization:</b> Implement Reserved Instances and right-sizing strategies<br/>
        5. <b>Monitoring Setup:</b> Establish comprehensive monitoring before migration begins
        """
        
        story.append(Paragraph(strategy_text, self.styles['Highlight']))
        
        return story

    def _create_enhanced_cost_analysis(self, analysis_results, analysis_mode):
        """Create enhanced cost analysis with charts"""
        story = []
        story.append(Paragraph("Cost Analysis & Financial Projections", self.styles['SectionHeader']))
        
        # Generate cost charts
        cost_charts = self.create_cost_comparison_chart(analysis_results, analysis_mode)
        
        # Add charts to story
        for chart_name, chart_buffer in cost_charts:
            chart_image = Image(chart_buffer, width=6*inch, height=3.6*inch)
            story.append(chart_image)
            story.append(Spacer(1, 0.2*inch))
        
        # Calculate total monthly cost for TCO
        if analysis_mode == 'single':
            valid_results = {k: v for k, v in analysis_results.items() if 'error' not in v}
            total_monthly = valid_results.get('PROD', {}).get('total_cost', 0) if valid_results else 0
        else:
            total_monthly = sum(r.get('PROD', {}).get('total_cost', 0) for r in analysis_results.values() 
                              if 'error' not in r and 'PROD' in r)
        
        # TCO Analysis Chart
        if total_monthly > 0:
            tco_chart = self.create_tco_analysis_chart(total_monthly)
            story.append(Paragraph("5-Year Total Cost of Ownership Analysis", self.styles['SubsectionHeader']))
            tco_image = Image(tco_chart, width=7*inch, height=3*inch)
            story.append(tco_image)
        
        return story

    def _create_enhanced_technical_specs(self, analysis_results, analysis_mode, server_specs):
        """Create enhanced technical specifications"""
        story = []
        story.append(Paragraph("Technical Specifications & Architecture", self.styles['SectionHeader']))
        
        if analysis_mode == 'single':
            story.append(Paragraph("Single Server Migration Specifications", self.styles['SubsectionHeader']))
            
            valid_results = {k: v for k, v in analysis_results.items() if 'error' not in v}
            
            for env, result in valid_results.items():
                story.append(Paragraph(f"{env} Environment Configuration", self.styles['SubsectionHeader']))
                
                if 'writer' in result:
                    # Aurora cluster
                    writer = result['writer']
                    config_data = [
                        ['Component', 'Specification', 'Performance Rationale'],
                        ['Aurora Writer', writer.get('instance_type', 'N/A'), 'Primary database operations'],
                        ['Writer vCPUs', str(writer.get('actual_vCPUs', 'N/A')), 'Compute capacity for workload'],
                        ['Writer RAM', f"{writer.get('actual_RAM_GB', 'N/A')} GB", 'In-memory caching and processing'],
                        ['Storage', f"{result.get('storage_GB', 'N/A')} GB", 'Auto-scaling Aurora storage'],
                        ['IOPS', f"{result.get('provisioned_iops', 'N/A'):,}", 'I/O performance guarantee']
                    ]
                    
                    if result.get('readers'):
                        for i, reader in enumerate(result['readers']):
                            config_data.append([f'Aurora Reader {i+1}', 
                                              reader.get('instance_type', 'N/A'), 
                                              'Read scaling and high availability'])
                else:
                    # Standard RDS
                    config_data = [
                        ['Component', 'Specification', 'Performance Rationale'],
                        ['RDS Instance', result.get('instance_type', 'N/A'), 'Compute and memory allocation'],
                        ['vCPUs', str(result.get('actual_vCPUs', 'N/A')), 'Processing power'],
                        ['RAM', f"{result.get('actual_RAM_GB', 'N/A')} GB", 'Database buffer pool'],
                        ['Storage', f"{result.get('storage_GB', 'N/A')} GB", 'Data and index storage'],
                        ['IOPS', f"{result.get('provisioned_iops', 'N/A'):,}", 'Disk I/O performance']
                    ]
                
                config_table = Table(config_data, colWidths=[1.5*inch, 2*inch, 2.5*inch])
                config_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#70ad47')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 11),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#e2efda')),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                story.append(config_table)
                story.append(Spacer(1, 0.2*inch))
        
        else:  # bulk analysis
            story.append(Paragraph("Bulk Migration Architecture Summary", self.styles['SubsectionHeader']))
            
            # Summary statistics
            total_servers = len(analysis_results)
            successful_analyses = sum(1 for result in analysis_results.values() if 'error' not in result)
            
            summary_text = f"""
            <b>Architecture Overview:</b><br/>
            • Total servers in scope: {total_servers}<br/>
            • Successfully analyzed: {successful_analyses}<br/>
            • Migration complexity: Mixed (heterogeneous and homogeneous)<br/>
            • Recommended approach: Phased migration with risk-based prioritization
            """
            
            story.append(Paragraph(summary_text, self.styles['Highlight']))
        
        return story

    def _create_enhanced_migration_strategy(self):
        """Create enhanced migration strategy with timeline chart"""
        story = []
        story.append(Paragraph("Migration Strategy & Timeline", self.styles['SectionHeader']))
        
        # Strategy overview
        strategy_text = """
        The migration strategy follows AWS Well-Architected Framework principles and incorporates industry 
        best practices for enterprise database migrations. Our phased approach minimizes risk while ensuring 
        business continuity throughout the transition.
        """
        story.append(Paragraph(strategy_text, self.styles['Normal']))
        story.append(Spacer(1, 0.2*inch))
        
        # Timeline chart
        timeline_chart = self.create_migration_timeline_chart()
        story.append(Paragraph("Migration Timeline & Phases", self.styles['SubsectionHeader']))
        timeline_image = Image(timeline_chart, width=7*inch, height=3*inch)
        story.append(timeline_image)
        story.append(Spacer(1, 0.2*inch))
        
        # Detailed phases
        phases_data = [
            ['Phase', 'Duration', 'Key Deliverables', 'Success Criteria'],
            ['Assessment & Planning', '2-3 weeks', 'Migration plan, risk assessment, team training', 'Complete inventory and roadmap'],
            ['Schema Conversion', '3-4 weeks', 'Converted schemas, SCT reports, compatibility analysis', '100% schema compatibility verified'],
            ['DMS Setup & Testing', '1-2 weeks', 'Replication instances, initial data sync', 'Successful test migration'],
            ['Application Testing', '4-5 weeks', 'Functional tests, performance validation', 'All test scenarios pass'],
            ['Production Cutover', '1 week', 'DNS cutover, final sync, go-live', 'Zero-downtime migration'],
            ['Optimization', '2-3 weeks', 'Performance tuning, monitoring setup', 'SLA compliance achieved']
        ]
        
        phases_table = Table(phases_data, colWidths=[1.3*inch, 1*inch, 2.2*inch, 1.5*inch])
        phases_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2e75b6')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#deebf7')),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('VALIGN', (0, 0), (-1, -1), 'TOP')
        ]))
        story.append(phases_table)
        
        return story

    def _create_enhanced_risk_assessment(self):
        """Create enhanced risk assessment with matrix visualization"""
        story = []
        story.append(Paragraph("Risk Assessment & Mitigation", self.styles['SectionHeader']))
        
        # Risk matrix chart
        risk_chart = self.create_risk_assessment_chart()
        story.append(Paragraph("Risk Assessment Matrix", self.styles['SubsectionHeader']))
        risk_image = Image(risk_chart, width=6*inch, height=4.8*inch)
        story.append(risk_image)
        story.append(Spacer(1, 0.2*inch))
        
        # Detailed risk table
        risk_data = [
            ['Risk Category', 'Probability', 'Impact', 'Mitigation Strategy', 'Owner'],
            ['Schema Conversion', 'Medium', 'High', 'AWS SCT + Expert Review + Parallel Testing', 'Database Team'],
            ['Performance Issues', 'Low', 'Medium', 'Load Testing + Baseline Monitoring + Tuning', 'Performance Team'],
            ['Data Corruption', 'Low', 'Critical', 'Checksums + Validation Scripts + Rollback Plan', 'Data Team'],
            ['Extended Downtime', 'Medium', 'High', 'Parallel Sync + Quick Cutover + Rehearsals', 'Migration Team'],
            ['Cost Overrun', 'Medium', 'Medium', 'Reserved Instances + Monitoring + Budget Alerts', 'Finance Team']
        ]
        
        risk_table = Table(risk_data, colWidths=[1.2*inch, 0.8*inch, 0.8*inch, 2.2*inch, 1*inch])
        risk_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#c5504b')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 9),
            ('FONTSIZE', (0, 1), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#fce4d6')),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('VALIGN', (0, 0), (-1, -1), 'TOP')
        ]))
        story.append(risk_table)
        
        return story

    def _create_enhanced_transfer_analysis(self, transfer_results):
        """Create enhanced data transfer analysis"""
        story = []
        story.append(Paragraph("Data Transfer Analysis", self.styles['SectionHeader']))
        
        # Transfer comparison chart
        transfer_methods = {}
        for method, result in transfer_results.items():
            transfer_methods[result.recommended_method[:20]] = result.total_cost
        
        if transfer_methods:
            transfer_chart = self.create_matplotlib_chart('bar', transfer_methods,
                                                        'Data Transfer Cost Comparison', figsize=(10, 5))
            transfer_image = Image(transfer_chart, width=6*inch, height=3*inch)
            story.append(transfer_image)
            story.append(Spacer(1, 0.2*inch))
        
        # Transfer details table
        transfer_data = [['Method', 'Transfer Time', 'Total Cost', 'Bandwidth Util.', 'Downtime', 'Recommendation']]
        
        for method, result in transfer_results.items():
            recommendation = "Recommended" if result.total_cost == min(r.total_cost for r in transfer_results.values()) else "Alternative"
            transfer_data.append([
                result.recommended_method[:25],
                f'{result.transfer_time_days:.1f} days',
                f'${result.total_cost:.2f}',
                f'{result.bandwidth_utilization:.0f}%',
                f'{result.estimated_downtime_hours:.1f}h',
                recommendation
            ])
        
        transfer_table = Table(transfer_data, colWidths=[1.5*inch, 0.8*inch, 0.8*inch, 0.8*inch, 0.6*inch, 0.9*inch])
        transfer_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#ffc000')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#1f4e79')),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 9),
            ('FONTSIZE', (0, 1), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#fff9e6')),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(transfer_table)
        
        return story

    def _create_enhanced_ai_insights(self, ai_insights):
        """Create enhanced AI insights section"""
        story = []
        story.append(Paragraph("AI-Powered Insights & Recommendations", self.styles['SectionHeader']))
        
        # AI summary box
        risk_level = ai_insights.get('risk_level', 'N/A')
        cost_opt = ai_insights.get('cost_optimization_potential', 0)
        
        ai_summary = f"""
        <b>AI ANALYSIS SUMMARY</b><br/><br/>
        • <b>Migration Risk Assessment:</b> {risk_level}<br/>
        • <b>Cost Optimization Potential:</b> {cost_opt * 100:.0f}%<br/>
        • <b>Confidence Level:</b> High (based on comprehensive analysis)<br/>
        • <b>Recommendation:</b> Proceed with structured migration approach
        """
        
        story.append(Paragraph(ai_summary, self.styles['KeyInsight']))
        story.append(Spacer(1, 0.2*inch))
        
        # Detailed AI analysis
        if 'ai_analysis' in ai_insights:
            story.append(Paragraph("Detailed AI Analysis", self.styles['SubsectionHeader']))
            
            ai_text = ai_insights['ai_analysis']
            # Split into paragraphs for better formatting
            paragraphs = ai_text.split('. ')
            
            current_para = ""
            for sentence in paragraphs:
                if len(current_para + sentence) > 500:
                    if current_para:
                        story.append(Paragraph(current_para.strip() + ".", self.styles['Normal']))
                        story.append(Spacer(1, 0.1*inch))
                    current_para = sentence
                else:
                    current_para += sentence + ". " if not sentence.endswith('.') else sentence + " "
            
            if current_para:
                story.append(Paragraph(current_para.strip(), self.styles['Normal']))
        
        return story

    def _create_implementation_roadmap(self):
        """Create implementation roadmap"""
        story = []
        story.append(Paragraph("Implementation Roadmap", self.styles['SectionHeader']))
        
        # Pre-migration checklist
        checklist_text = """
        <b>PRE-MIGRATION CHECKLIST</b><br/><br/>
        ✓ Executive approval and budget allocation<br/>
        ✓ Technical team training and AWS certification<br/>
        ✓ Network connectivity (Direct Connect) setup<br/>
        ✓ Security and compliance review completed<br/>
        ✓ Backup and disaster recovery procedures validated<br/>
        ✓ Performance baseline and monitoring tools configured
        """
        
        story.append(Paragraph(checklist_text, self.styles['Highlight']))
        story.append(Spacer(1, 0.2*inch))
        
        # Success metrics
        success_data = [
            ['Metric', 'Target', 'Measurement Method'],
            ['Migration Downtime', '< 4 hours', 'Planned maintenance window tracking'],
            ['Performance Degradation', '< 10%', 'APM tools and baseline comparison'],
            ['Data Integrity', '100%', 'Checksum validation and row counts'],
            ['Cost Variance', '< 15%', 'Monthly AWS billing analysis'],
            ['Team Productivity', '> 95%', 'Developer surveys and incident metrics'],
            ['Go-Live Success Rate', '100%', 'All systems operational post-migration']
        ]
        
        success_table = Table(success_data, colWidths=[2*inch, 1.5*inch, 2.5*inch])
        success_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#70ad47')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#e2efda')),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(Paragraph("Success Metrics & KPIs", self.styles['SubsectionHeader']))
        story.append(success_table)
        
        return story

    def _create_appendices(self, analysis_results, analysis_mode):
        """Create appendices with additional details"""
        story = []
        story.append(Paragraph("Appendices", self.styles['SectionHeader']))
        
        # Appendix A: Detailed cost breakdown
        story.append(Paragraph("Appendix A: Detailed Cost Breakdown", self.styles['SubsectionHeader']))
        
        if analysis_mode == 'single':
            valid_results = {k: v for k, v in analysis_results.items() if 'error' not in v}
            
            for env, result in valid_results.items():
                cost_breakdown_data = [['Cost Component', 'Monthly Cost', 'Annual Cost', 'Percentage']]
                
                total_cost = result.get('total_cost', 0)
                
                if 'writer' in result:
                    writer_cost = result.get('cost_breakdown', {}).get('writer_monthly', 0)
                    readers_cost = result.get('cost_breakdown', {}).get('readers_monthly', 0)
                    storage_cost = result.get('cost_breakdown', {}).get('storage_monthly', 0)
                    backup_cost = result.get('cost_breakdown', {}).get('backup_monthly', 0)
                    
                    cost_breakdown_data.extend([
                        ['Aurora Writer Instance', f'${writer_cost:.2f}', f'${writer_cost * 12:.2f}', f'{(writer_cost/total_cost)*100:.1f}%'],
                        ['Aurora Reader Instances', f'${readers_cost:.2f}', f'${readers_cost * 12:.2f}', f'{(readers_cost/total_cost)*100:.1f}%'],
                        ['Aurora Storage', f'${storage_cost:.2f}', f'${storage_cost * 12:.2f}', f'{(storage_cost/total_cost)*100:.1f}%'],
                        ['Backup Storage', f'${backup_cost:.2f}', f'${backup_cost * 12:.2f}', f'{(backup_cost/total_cost)*100:.1f}%']
                    ])
                else:
                    instance_cost = result.get('instance_cost', 0)
                    storage_cost = result.get('storage_cost', 0)
                    
                    cost_breakdown_data.extend([
                        ['RDS Instance', f'${instance_cost:.2f}', f'${instance_cost * 12:.2f}', f'{(instance_cost/total_cost)*100:.1f}%'],
                        ['RDS Storage', f'${storage_cost:.2f}', f'${storage_cost * 12:.2f}', f'{(storage_cost/total_cost)*100:.1f}%']
                    ])
                
                cost_table = Table(cost_breakdown_data, colWidths=[2*inch, 1.2*inch, 1.2*inch, 1*inch])
                cost_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2e75b6')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 10),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#deebf7')),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                story.append(Paragraph(f"{env} Environment Cost Details", self.styles['Normal']))
                story.append(cost_table)
                story.append(Spacer(1, 0.2*inch))
        
        # Appendix B: Technical assumptions
        story.append(Paragraph("Appendix B: Technical Assumptions", self.styles['SubsectionHeader']))
        
        assumptions_text = """
        • Instance sizing includes 20% buffer for peak workloads<br/>
        • Storage growth calculated at 20% annually<br/>
        • Network transfer assumes encrypted connections<br/>
        • Backup retention set to 7 days automated + 35 days snapshot<br/>
        • Multi-AZ deployment for production environments<br/>
        • Performance Insights enabled for monitoring<br/>
        • Enhanced monitoring at 60-second intervals
        """
        
        story.append(Paragraph(assumptions_text, self.styles['Normal']))
        
        return story

# Helper function to integrate with the Streamlit app
def generate_super_enhanced_pdf_report(analysis_results, analysis_mode, server_specs=None, 
                                     ai_insights=None, transfer_results=None):
    """
    Generate the super enhanced PDF report with graphs and visualizations
    """
    try:
        generator = SuperEnhancedReportGenerator()
        pdf_bytes = generator.generate_comprehensive_pdf_report(
            analysis_results=analysis_results,
            analysis_mode=analysis_mode,
            server_specs=server_specs,
            ai_insights=ai_insights,
            transfer_results=transfer_results
        )
        return pdf_bytes
    except Exception as e:
        print(f"Error generating super enhanced PDF: {e}")
        import traceback
        traceback.print_exc()
        return None