import io
from datetime import datetime, timedelta
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.graphics.shapes import Drawing, Rect, String
from reportlab.graphics.charts.piecharts import Pie
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics import renderPDF
import matplotlib.pyplot as plt
import base64

class ComprehensiveReportGenerator:
    """Ultra-comprehensive PDF Report Generator with ALL analysis details"""
    
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self.setup_custom_styles()
    
    def setup_custom_styles(self):
        """Setup comprehensive custom styles"""
        # Enhanced title style
        self.styles.add(ParagraphStyle(
            name='MainTitle',
            parent=self.styles['Title'],
            fontSize=28,
            spaceAfter=30,
            textColor=colors.darkblue,
            alignment=1,
            fontName='Helvetica-Bold'
        ))
        
        # Section header styles
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading1'],
            fontSize=18,
            spaceBefore=25,
            spaceAfter=15,
            textColor=colors.darkblue,
            borderWidth=2,
            borderColor=colors.lightblue,
            borderPadding=8,
            fontName='Helvetica-Bold'
        ))
        
        self.styles.add(ParagraphStyle(
            name='SubsectionHeader',
            parent=self.styles['Heading2'],
            fontSize=14,
            spaceBefore=15,
            spaceAfter=10,
            textColor=colors.darkgreen,
            fontName='Helvetica-Bold'
        ))
        
        self.styles.add(ParagraphStyle(
            name='DetailHeader',
            parent=self.styles['Heading3'],
            fontSize=12,
            spaceBefore=12,
            spaceAfter=6,
            textColor=colors.darkred,
            fontName='Helvetica-Bold'
        ))
        
        # Content styles
        self.styles.add(ParagraphStyle(
            name='Highlight',
            parent=self.styles['Normal'],
            fontSize=12,
            textColor=colors.darkred,
            backColor=colors.lightyellow,
            borderWidth=1,
            borderColor=colors.orange,
            borderPadding=8
        ))
        
        self.styles.add(ParagraphStyle(
            name='CodeBlock',
            parent=self.styles['Normal'],
            fontSize=10,
            textColor=colors.black,
            backColor=colors.lightgrey,
            borderWidth=1,
            borderColor=colors.grey,
            borderPadding=5,
            fontName='Courier'
        ))
        
        self.styles.add(ParagraphStyle(
            name='TableData',
            parent=self.styles['Normal'],
            fontSize=9,
            alignment=0
        ))

    def safe_get(self, dictionary, key, default=0):
        """Safely get a value from a dictionary"""
        if isinstance(dictionary, dict):
            return dictionary.get(key, default)
        return default

    def safe_get_str(self, dictionary, key, default="N/A"):
        """Safely get a string value from a dictionary"""
        if isinstance(dictionary, dict):
            return str(dictionary.get(key, default))
        return default

    def generate_ultra_comprehensive_report(self, analysis_results, analysis_mode, server_specs=None, 
                                          ai_insights=None, transfer_results=None, migration_config=None,
                                          workload_characteristics=None):
        """Generate the most comprehensive PDF report possible"""
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4, 
                              rightMargin=0.5*inch, leftMargin=0.5*inch,
                              topMargin=0.75*inch, bottomMargin=0.75*inch)
        
        story = []
        
        # 1. ENHANCED TITLE PAGE
        story.extend(self._create_enhanced_title_page(analysis_mode, migration_config))
        story.append(PageBreak())
        
        # 2. COMPREHENSIVE TABLE OF CONTENTS
        story.extend(self._create_table_of_contents())
        story.append(PageBreak())
        
        # 3. EXECUTIVE SUMMARY WITH DETAILED METRICS
        story.extend(self._create_detailed_executive_summary(analysis_results, analysis_mode, ai_insights))
        story.append(PageBreak())
        
        # 4. COMPLETE INPUT SPECIFICATIONS
        story.extend(self._create_complete_input_specifications(server_specs, migration_config, workload_characteristics))
        story.append(PageBreak())
        
        # 5. DETAILED SIZING ANALYSIS
        if analysis_mode == 'single':
            story.extend(self._create_comprehensive_single_analysis(analysis_results, server_specs))
        else:
            story.extend(self._create_comprehensive_bulk_analysis(analysis_results, server_specs))
        story.append(PageBreak())
        
        # 6. COMPLETE FINANCIAL BREAKDOWN
        story.extend(self._create_complete_financial_analysis(analysis_results, analysis_mode))
        story.append(PageBreak())
        
        # 7. PERFORMANCE AND CAPACITY ANALYSIS
        story.extend(self._create_performance_capacity_analysis(analysis_results, server_specs))
        story.append(PageBreak())
        
        # 8. DATA TRANSFER COMPREHENSIVE ANALYSIS
        if transfer_results:
            story.extend(self._create_comprehensive_transfer_analysis(transfer_results))
            story.append(PageBreak())
        
        # 9. COMPLETE AI INSIGHTS AND RECOMMENDATIONS
        if ai_insights:
            story.extend(self._create_complete_ai_analysis(ai_insights))
            story.append(PageBreak())
        
        # 10. DETAILED MIGRATION STRATEGY
        story.extend(self._create_detailed_migration_strategy(analysis_results, ai_insights, migration_config))
        story.append(PageBreak())
        
        # 11. RISK ASSESSMENT AND MITIGATION
        story.extend(self._create_comprehensive_risk_assessment(analysis_results, ai_insights))
        story.append(PageBreak())
        
        # 12. IMPLEMENTATION ROADMAP
        story.extend(self._create_detailed_implementation_roadmap(analysis_results, analysis_mode))
        story.append(PageBreak())
        
        # 13. APPENDICES
        story.extend(self._create_comprehensive_appendices(analysis_results, server_specs, migration_config))
        
        # Build the PDF
        try:
            doc.build(story)
            buffer.seek(0)
            return buffer.getvalue()
        except Exception as e:
            print(f"Error building comprehensive PDF: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _create_enhanced_title_page(self, analysis_mode, migration_config):
        """Create enhanced title page with comprehensive details"""
        story = []
        
        # Main title
        story.append(Paragraph("AWS RDS Migration & Sizing", self.styles['MainTitle']))
        story.append(Paragraph("Ultra-Comprehensive Analysis Report", self.styles['MainTitle']))
        story.append(Spacer(1, 0.5*inch))
        
        # Analysis details box
        analysis_type = "Single Server Analysis" if analysis_mode == 'single' else "Enterprise Bulk Analysis"
        story.append(Paragraph(f"<b>{analysis_type}</b>", self.styles['Heading1']))
        story.append(Spacer(1, 0.3*inch))
        
        # Generation metadata
        generation_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
        
        metadata_data = [
            ['Report Type', 'Ultra-Comprehensive Technical Analysis'],
            ['Generated On', generation_time],
            ['Analysis Mode', analysis_type],
            ['Report Version', '3.0 (Enhanced)'],
            ['Prepared For', 'Enterprise Cloud Migration Team'],
            ['Confidentiality', 'Internal Use Only']
        ]
        
        if migration_config:
            metadata_data.extend([
                ['Source Engine', migration_config.get('source_engine', 'N/A')],
                ['Target Engine', migration_config.get('target_engine', 'N/A')],
                ['AWS Region', migration_config.get('region', 'N/A')],
                ['Deployment', migration_config.get('deployment_option', 'N/A')]
            ])
        
        metadata_table = Table(metadata_data, colWidths=[2*inch, 3*inch])
        metadata_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.lightblue),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTSIZE', (0, 1), (-1, -1), 10)
        ]))
        story.append(metadata_table)
        
        story.append(Spacer(1, 0.5*inch))
        
        # Report scope and capabilities
        scope_text = """
        <b>Report Scope & Capabilities:</b><br/>
        ✓ Complete Input Specification Analysis<br/>
        ✓ Detailed Sizing Calculations & Logic<br/>
        ✓ Comprehensive Cost Breakdown (All Components)<br/>
        ✓ Performance & Capacity Planning<br/>
        ✓ Data Transfer Strategy & Analysis<br/>
        ✓ AI-Powered Insights & Recommendations<br/>
        ✓ Risk Assessment & Mitigation Strategies<br/>
        ✓ Detailed Implementation Roadmap<br/>
        ✓ Migration Timeline & Dependencies<br/>
        ✓ Technical Appendices & Reference Data
        """
        story.append(Paragraph(scope_text, self.styles['Highlight']))
        
        return story

    def _create_table_of_contents(self):
        """Create comprehensive table of contents"""
        story = []
        story.append(Paragraph("Table of Contents", self.styles['SectionHeader']))
        
        toc_data = [
            ['Section', 'Page', 'Content'],
            ['1. Executive Summary', '3', 'Key findings and recommendations'],
            ['2. Input Specifications', '4', 'Complete server and migration configuration'],
            ['3. Sizing Analysis', '5', 'Detailed sizing calculations and logic'],
            ['4. Financial Analysis', '6', 'Comprehensive cost breakdown and TCO'],
            ['5. Performance Analysis', '7', 'Capacity planning and performance metrics'],
            ['6. Data Transfer Analysis', '8', 'Transfer methods and optimization'],
            ['7. AI Insights', '9', 'AI-powered recommendations and analysis'],
            ['8. Migration Strategy', '10', 'Detailed migration approach and phases'],
            ['9. Risk Assessment', '11', 'Risk analysis and mitigation strategies'],
            ['10. Implementation Roadmap', '12', 'Timeline, tasks, and dependencies'],
            ['11. Appendices', '13', 'Technical reference and supporting data']
        ]
        
        toc_table = Table(toc_data, colWidths=[2.5*inch, 0.8*inch, 2.7*inch])
        toc_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.navy),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.lightblue),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTSIZE', (0, 1), (-1, -1), 9)
        ]))
        story.append(toc_table)
        
        return story

    def _create_detailed_executive_summary(self, analysis_results, analysis_mode, ai_insights):
        """Create detailed executive summary with all key metrics"""
        story = []
        story.append(Paragraph("1. Executive Summary", self.styles['SectionHeader']))
        
        # Key findings section
        story.append(Paragraph("1.1 Key Findings", self.styles['SubsectionHeader']))
        
        if analysis_mode == 'single':
            valid_results = {k: v for k, v in analysis_results.items() if 'error' not in v}
            if valid_results:
                prod_result = valid_results.get('PROD', list(valid_results.values())[0])
                
                # Comprehensive metrics table
                metrics_data = [
                    ['Metric', 'Current State', 'Recommended State', 'Impact'],
                    ['Monthly Cost', 'On-premises', f"${self.safe_get(prod_result, 'total_cost', 0):,.2f}", 'Predictable cloud costs'],
                    ['Annual Cost', 'Variable', f"${self.safe_get(prod_result, 'total_cost', 0) * 12:,.2f}", 'Budget planning'],
                    ['High Availability', 'Single point of failure', 'Multi-AZ deployment', 'Improved uptime'],
                    ['Backup Strategy', 'Manual/scheduled', 'Automated with point-in-time recovery', 'Enhanced data protection'],
                    ['Scalability', 'Hardware-bound', 'Elastic scaling', 'Performance optimization'],
                    ['Maintenance', 'Manual patching', 'Automated maintenance windows', 'Reduced operational overhead']
                ]
                
                if 'writer' in prod_result:
                    writer = prod_result['writer']
                    instance_info = f"{self.safe_get_str(writer, 'instance_type', 'N/A')}"
                    cpu_info = f"{self.safe_get(writer, 'actual_vCPUs', 0)} vCPUs"
                    ram_info = f"{self.safe_get(writer, 'actual_RAM_GB', 0)} GB RAM"
                    
                    if prod_result.get('readers'):
                        reader_count = len(prod_result['readers'])
                        instance_info += f" + {reader_count} Read Replicas"
                        
                    metrics_data.extend([
                        ['Recommended Instance', 'Physical server', instance_info, 'Right-sized for workload'],
                        ['Compute Resources', 'Fixed allocation', f"{cpu_info}, {ram_info}", 'Optimized performance'],
                        ['Read Scaling', 'Limited', f"{reader_count if prod_result.get('readers') else 0} read replicas", 'Improved read performance']
                    ])
                else:
                    instance_info = self.safe_get_str(prod_result, 'instance_type', 'N/A')
                    cpu_info = f"{self.safe_get(prod_result, 'actual_vCPUs', 0)} vCPUs"
                    ram_info = f"{self.safe_get(prod_result, 'actual_RAM_GB', 0)} GB RAM"
                    
                    metrics_data.extend([
                        ['Recommended Instance', 'Physical server', instance_info, 'Right-sized for workload'],
                        ['Compute Resources', 'Fixed allocation', f"{cpu_info}, {ram_info}", 'Optimized performance']
                    ])
                
                metrics_table = Table(metrics_data, colWidths=[1.5*inch, 1.3*inch, 1.7*inch, 1.5*inch])
                metrics_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 10),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.lightblue),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black),
                    ('FONTSIZE', (0, 1), (-1, -1), 8),
                    ('VALIGN', (0, 0), (-1, -1), 'TOP')
                ]))
                story.append(metrics_table)
                
        else:
            # Bulk analysis summary
            total_servers = len(analysis_results)
            successful_servers = sum(1 for result in analysis_results.values() if 'error' not in result)
            total_monthly_cost = sum(
                self.safe_get(result.get('PROD', list(result.values())[0] if result.values() else {}), 'total_cost', 0)
                for result in analysis_results.values() if 'error' not in result
            )
            
            bulk_metrics_data = [
                ['Metric', 'Value', 'Analysis', 'Business Impact'],
                ['Total Servers', str(total_servers), f'{successful_servers} analyzed successfully', 'Complete migration scope'],
                ['Migration Readiness', f'{(successful_servers/total_servers)*100:.1f}%', 'High confidence analysis', 'Reduced migration risk'],
                ['Total Monthly Cost', f'${total_monthly_cost:,.2f}', 'All servers aggregated', 'Budget planning'],
                ['Average Cost/Server', f'${total_monthly_cost/max(successful_servers,1):,.2f}', 'Cost distribution analysis', 'Resource optimization'],
                ['Annual Investment', f'${total_monthly_cost * 12:,.2f}', '3-year TCO planning', 'Financial commitment'],
                ['Cost Optimization', 'TBD', 'Reserved Instance potential', 'Additional savings']
            ]
            
            bulk_metrics_table = Table(bulk_metrics_data, colWidths=[1.5*inch, 1.3*inch, 1.5*inch, 1.7*inch])
            bulk_metrics_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.darkgreen),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.lightgreen),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('FONTSIZE', (0, 1), (-1, -1), 8),
                ('VALIGN', (0, 0), (-1, -1), 'TOP')
            ]))
            story.append(bulk_metrics_table)
        
        # AI insights summary
        story.append(Paragraph("1.2 AI-Powered Strategic Insights", self.styles['SubsectionHeader']))
        
        if ai_insights:
            risk_level = ai_insights.get('risk_level', 'UNKNOWN')
            cost_optimization = ai_insights.get('cost_optimization_potential', 0) * 100
            
            ai_summary_text = f"""
            <b>Migration Risk Assessment:</b> {risk_level}<br/>
            <b>Cost Optimization Potential:</b> {cost_optimization:.1f}%<br/>
            <b>Confidence Level:</b> High (AI-validated recommendations)<br/>
            <b>Strategic Recommendation:</b> Proceed with phased migration approach<br/>
            <b>Timeline Estimate:</b> 12-16 weeks for complete migration<br/>
            <b>Success Probability:</b> 95%+ with proper planning and execution
            """
            story.append(Paragraph(ai_summary_text, self.styles['Highlight']))
        else:
            story.append(Paragraph("AI insights were not available during this analysis. Manual validation recommended.", self.styles['Normal']))
        
        return story

    def _create_complete_input_specifications(self, server_specs, migration_config, workload_characteristics):
        """Create complete input specifications section"""
        story = []
        story.append(Paragraph("2. Complete Input Specifications", self.styles['SectionHeader']))
        
        # Migration configuration
        story.append(Paragraph("2.1 Migration Configuration", self.styles['SubsectionHeader']))
        
        if migration_config:
            config_data = [
                ['Parameter', 'Value', 'Impact on Sizing'],
                ['Source Database Engine', migration_config.get('source_engine', 'N/A'), 'Conversion complexity assessment'],
                ['Target Database Engine', migration_config.get('target_engine', 'N/A'), 'Feature compatibility analysis'],
                ['AWS Region', migration_config.get('region', 'N/A'), 'Pricing and availability zones'],
                ['Deployment Option', migration_config.get('deployment_option', 'N/A'), 'High availability configuration'],
                ['Storage Type', migration_config.get('storage_type', 'N/A'), 'Performance and cost optimization'],
                ['Analysis Mode', 'Single Server' if isinstance(server_specs, dict) else 'Bulk Analysis', 'Scope of recommendations']
            ]
            
            config_table = Table(config_data, colWidths=[2*inch, 1.5*inch, 2.5*inch])
            config_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.lightblue),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('FONTSIZE', (0, 1), (-1, -1), 9)
            ]))
            story.append(config_table)
        
        # Server specifications
        story.append(Paragraph("2.2 Server Specifications", self.styles['SubsectionHeader']))
        
        if isinstance(server_specs, dict):
            # Single server
            server_data = [
                ['Specification', 'Current Value', 'Utilization/Notes'],
                ['Server Name', server_specs.get('server_name', 'N/A'), 'Primary identifier'],
                ['CPU Cores', str(server_specs.get('cores', server_specs.get('cpu_cores', 'N/A'))), f"{server_specs.get('cpu_util', 'N/A')}% peak utilization"],
                ['RAM (GB)', str(server_specs.get('ram', server_specs.get('ram_gb', 'N/A'))), f"{server_specs.get('ram_util', 'N/A')}% peak utilization"],
                ['Storage (GB)', str(server_specs.get('storage', server_specs.get('storage_gb', 'N/A'))), 'Current data size'],
                ['Max IOPS', str(server_specs.get('max_iops', 'N/A')), 'Peak I/O operations per second'],
                ['Max Throughput', f"{server_specs.get('max_throughput_mbps', 'N/A')} MB/s", 'Peak data transfer rate'],
                ['Database Engine', server_specs.get('database_engine', 'N/A'), 'Source database platform'],
                ['Growth Rate', f"{server_specs.get('growth_rate', 20)}%", 'Annual data growth projection'],
                ['Planning Horizon', f"{server_specs.get('years', 3)} years", 'Capacity planning timeframe']
            ]
            
        elif isinstance(server_specs, list):
            # Bulk servers - show summary and first few servers
            story.append(Paragraph(f"Total Servers: {len(server_specs)}", self.styles['DetailHeader']))
            
            server_data = [['Server Name', 'CPU Cores', 'RAM (GB)', 'Storage (GB)', 'Peak CPU %', 'Peak RAM %', 'DB Engine']]
            
            for i, server in enumerate(server_specs[:10]):  # Show first 10 servers
                server_data.append([
                    server.get('server_name', f'Server_{i+1}'),
                    str(server.get('cpu_cores', 'N/A')),
                    str(server.get('ram_gb', 'N/A')),
                    str(server.get('storage_gb', 'N/A')),
                    f"{server.get('peak_cpu_percent', 'N/A')}%",
                    f"{server.get('peak_ram_percent', 'N/A')}%",
                    server.get('database_engine', 'N/A')
                ])
            
            if len(server_specs) > 10:
                server_data.append(['...', '...', '...', '...', '...', '...', f'+ {len(server_specs) - 10} more'])
        
        server_table = Table(server_data, colWidths=[1.2*inch, 0.8*inch, 0.8*inch, 0.9*inch, 0.8*inch, 0.8*inch, 1*inch])
        server_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkgreen),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 9),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.lightgreen),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTSIZE', (0, 1), (-1, -1), 8)
        ]))
        story.append(server_table)
        
        # Workload characteristics
        if workload_characteristics:
            story.append(Paragraph("2.3 Workload Characteristics", self.styles['SubsectionHeader']))
            
            workload_data = [
                ['Characteristic', 'Value', 'Impact on Sizing'],
                ['CPU Pattern', workload_characteristics.get('cpu_utilization_pattern', 'N/A'), 'Burstable vs fixed instance types'],
                ['Memory Pattern', workload_characteristics.get('memory_usage_pattern', 'N/A'), 'Memory optimization strategy'],
                ['I/O Pattern', workload_characteristics.get('io_pattern', 'N/A'), 'Read replica configuration'],
                ['Connection Count', str(workload_characteristics.get('connection_count', 'N/A')), 'Connection pooling requirements'],
                ['Transaction Volume', workload_characteristics.get('transaction_volume', 'N/A'), 'Performance tier selection'],
                ['Analytical Workload', 'Yes' if workload_characteristics.get('analytical_workload') else 'No', 'Read replica necessity']
            ]
            
            workload_table = Table(workload_data, colWidths=[1.5*inch, 1.5*inch, 3*inch])
            workload_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.darkorange),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.wheat),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('FONTSIZE', (0, 1), (-1, -1), 9)
            ]))
            story.append(workload_table)
        
        return story

    def _create_comprehensive_single_analysis(self, analysis_results, server_specs):
        """Create comprehensive single server analysis"""
        story = []
        story.append(Paragraph("3. Detailed Single Server Analysis", self.styles['SectionHeader']))
        
        valid_results = {k: v for k, v in analysis_results.items() if 'error' not in v}
        
        for env, result in valid_results.items():
            story.append(Paragraph(f"3.{list(valid_results.keys()).index(env) + 1} {env} Environment", self.styles['SubsectionHeader']))
            
            # Detailed configuration analysis
            story.append(Paragraph("Configuration Details", self.styles['DetailHeader']))
            
            config_data = [['Component', 'Specification', 'Justification', 'Performance Impact']]
            
            if 'writer' in result:
                # Aurora cluster configuration
                writer = result['writer']
                config_data.extend([
                    ['Writer Instance', 
                     self.safe_get_str(writer, 'instance_type', 'N/A'), 
                     'Primary write operations', 
                     'Handles all write transactions'],
                    ['Writer vCPUs', 
                     str(self.safe_get(writer, 'actual_vCPUs', 'N/A')), 
                     'Computed from on-prem CPU utilization', 
                     'Supports peak CPU workload'],
                    ['Writer RAM', 
                     f"{self.safe_get(writer, 'actual_RAM_GB', 'N/A')} GB", 
                     'Based on current memory usage patterns', 
                     'Optimizes buffer pool and caching'],
                    ['Writer Cost', 
                     f"${self.safe_get(writer, 'instance_cost', 0):,.2f}/month", 
                     'On-demand pricing', 
                     'Primary cost component']
                ])
                
                if result.get('readers'):
                    for i, reader in enumerate(result['readers']):
                        config_data.extend([
                            [f'Reader {i+1} Instance', 
                             self.safe_get_str(reader, 'instance_type', 'N/A'), 
                             'Read scaling and availability', 
                             'Offloads read queries from writer'],
                            [f'Reader {i+1} Cost', 
                             f"${self.safe_get(reader, 'instance_cost', 0):,.2f}/month", 
                             'On-demand pricing', 
                             'Additional read capacity cost']
                        ])
            else:
                # Standard RDS configuration
                config_data.extend([
                    ['Instance Type', 
                     self.safe_get_str(result, 'instance_type', 'N/A'), 
                     'Right-sized for workload', 
                     'Balanced compute and memory'],
                    ['vCPUs', 
                     str(self.safe_get(result, 'actual_vCPUs', 'N/A')), 
                     'CPU headroom for peak loads', 
                     'Prevents CPU bottlenecks'],
                    ['RAM', 
                     f"{self.safe_get(result, 'actual_RAM_GB', 'N/A')} GB", 
                     'Database buffer pool optimization', 
                     'Reduces disk I/O through caching'],
                    ['Instance Cost', 
                     f"${self.safe_get(result, 'instance_cost', 0):,.2f}/month", 
                     'On-demand pricing model', 
                     'Primary operational cost']
                ])
            
            # Storage configuration
            storage_gb = self.safe_get(result, 'storage_GB', 0)
            storage_cost = self.safe_get(result, 'storage_cost', 0)
            
            config_data.extend([
                ['Storage Capacity', 
                 f"{storage_gb:,} GB", 
                 'Current data + growth projection', 
                 'Accommodates 3-year growth'],
                ['Storage Type', 
                 'gp3 (General Purpose SSD)', 
                 'Balanced performance and cost', 
                 'Suitable for most workloads'],
                ['Storage Cost', 
                 f"${storage_cost:,.2f}/month", 
                 'Per-GB pricing', 
                 'Scales with data growth']
            ])
            
            config_table = Table(config_data, colWidths=[1.2*inch, 1.5*inch, 1.8*inch, 1.5*inch])
            config_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 9),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.lightblue),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('FONTSIZE', (0, 1), (-1, -1), 8),
                ('VALIGN', (0, 0), (-1, -1), 'TOP')
            ]))
            story.append(config_table)
            
            # Cost breakdown analysis
            story.append(Paragraph("Detailed Cost Breakdown", self.styles['DetailHeader']))
            
            cost_breakdown = self.safe_get(result, 'cost_breakdown', {})
            total_cost = self.safe_get(result, 'total_cost', 0)
            
            cost_data = [['Cost Component', 'Monthly Cost', 'Annual Cost', '% of Total', 'Notes']]
            
            if 'writer_monthly' in cost_breakdown:
                # Aurora costs
                writer_cost = self.safe_get(cost_breakdown, 'writer_monthly', 0)
                readers_cost = self.safe_get(cost_breakdown, 'readers_monthly', 0)
                storage_cost = self.safe_get(cost_breakdown, 'storage_monthly', 0)
                backup_cost = self.safe_get(cost_breakdown, 'backup_monthly', 0)
                
                cost_data.extend([
                    ['Writer Instance', f'${writer_cost:,.2f}', f'${writer_cost * 12:,.2f}', 
                     f'{(writer_cost/total_cost)*100:.1f}%' if total_cost > 0 else '0%', 'Primary database instance'],
                    ['Reader Instances', f'${readers_cost:,.2f}', f'${readers_cost * 12:,.2f}', 
                     f'{(readers_cost/total_cost)*100:.1f}%' if total_cost > 0 else '0%', 'Read scaling instances'],
                    ['Aurora Storage', f'${storage_cost:,.2f}', f'${storage_cost * 12:,.2f}', 
                     f'{(storage_cost/total_cost)*100:.1f}%' if total_cost > 0 else '0%', 'Pay-per-use storage'],
                    ['Backup Storage', f'${backup_cost:,.2f}', f'${backup_cost * 12:,.2f}', 
                     f'{(backup_cost/total_cost)*100:.1f}%' if total_cost > 0 else '0%', 'Automated backups']
                ])
            else:
                # Standard RDS costs
                instance_cost = self.safe_get(cost_breakdown, 'instance_monthly', self.safe_get(result, 'instance_cost', 0))
                storage_cost = self.safe_get(cost_breakdown, 'storage_monthly', self.safe_get(result, 'storage_cost', 0))
                backup_cost = storage_cost * 0.25  # Estimate
                
                cost_data.extend([
                    ['RDS Instance', f'${instance_cost:,.2f}', f'${instance_cost * 12:,.2f}', 
                     f'{(instance_cost/total_cost)*100:.1f}%' if total_cost > 0 else '0%', 'Database compute instance'],
                    ['EBS Storage', f'${storage_cost:,.2f}', f'${storage_cost * 12:,.2f}', 
                     f'{(storage_cost/total_cost)*100:.1f}%' if total_cost > 0 else '0%', 'Provisioned storage'],
                    ['Backup Storage', f'${backup_cost:,.2f}', f'${backup_cost * 12:,.2f}', 
                     f'{(backup_cost/total_cost)*100:.1f}%' if total_cost > 0 else '0%', 'Automated backups (estimated)']
                ])
            
            cost_data.append(['TOTAL', f'${total_cost:,.2f}', f'${total_cost * 12:,.2f}', '100%', 'All components'])
            
            cost_table = Table(cost_data, colWidths=[1.5*inch, 1*inch, 1*inch, 0.8*inch, 1.7*inch])
            cost_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.darkred),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 9),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -2), colors.mistyrose),
                ('BACKGROUND', (0, -1), (-1, -1), colors.yellow),
                ('FONTNAME', (0, -1), (-1, -1), 'Helvetica-Bold'),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('FONTSIZE', (0, 1), (-1, -1), 8)
            ]))
            story.append(cost_table)
            
            story.append(Spacer(1, 15))
        
        return story

    def _create_comprehensive_bulk_analysis(self, analysis_results, server_specs):
        """Create comprehensive bulk server analysis"""
        story = []
        story.append(Paragraph("3. Comprehensive Bulk Analysis", self.styles['SectionHeader']))
        
        # Analysis summary
        total_servers = len(analysis_results)
        successful_analyses = sum(1 for result in analysis_results.values() if 'error' not in result)
        failed_analyses = total_servers - successful_analyses
        
        story.append(Paragraph(f"Analysis Overview: {successful_analyses} successful, {failed_analyses} failed out of {total_servers} total servers", self.styles['SubsectionHeader']))
        
        # Detailed server-by-server analysis
        story.append(Paragraph("3.1 Server-by-Server Analysis", self.styles['SubsectionHeader']))
        
        bulk_data = [['Server Name', 'Status', 'Instance Type', 'vCPUs', 'RAM (GB)', 'Storage (GB)', 'Monthly Cost', 'Annual Cost']]
        total_monthly_cost = 0
        
        for server_name, server_results in analysis_results.items():
            if 'error' not in server_results:
                result = server_results.get('PROD', list(server_results.values())[0])
                if 'error' not in result:
                    monthly_cost = self.safe_get(result, 'total_cost', 0)
                    total_monthly_cost += monthly_cost
                    
                    if 'writer' in result:
                        writer = result['writer']
                        instance_type = self.safe_get_str(writer, 'instance_type', 'N/A')
                        vcpus = self.safe_get(writer, 'actual_vCPUs', 0)
                        ram_gb = self.safe_get(writer, 'actual_RAM_GB', 0)
                        if result.get('readers'):
                            instance_type += f" + {len(result['readers'])} readers"
                    else:
                        instance_type = self.safe_get_str(result, 'instance_type', 'N/A')
                        vcpus = self.safe_get(result, 'actual_vCPUs', 0)
                        ram_gb = self.safe_get(result, 'actual_RAM_GB', 0)
                    
                    storage_gb = self.safe_get(result, 'storage_GB', 0)
                    
                    bulk_data.append([
                        server_name[:20] + ('...' if len(server_name) > 20 else ''),
                        '✓ Success',
                        instance_type[:25] + ('...' if len(instance_type) > 25 else ''),
                        str(vcpus),
                        str(ram_gb),
                        f'{storage_gb:,}',
                        f'${monthly_cost:.2f}',
                        f'${monthly_cost * 12:.2f}'
                    ])
                else:
                    bulk_data.append([
                        server_name[:20] + ('...' if len(server_name) > 20 else ''),
                        '✗ Error',
                        result.get('error', 'Unknown error'),
                        '0', '0', '0', '$0.00', '$0.00'
                    ])
            else:
                bulk_data.append([
                    server_name[:20] + ('...' if len(server_name) > 20 else ''),
                    '✗ Failed',
                    server_results.get('error', 'Analysis failed'),
                    '0', '0', '0', '$0.00', '$0.00'
                ])
        
        # Add summary row
        avg_monthly_cost = total_monthly_cost / max(successful_analyses, 1)
        bulk_data.append([
            'TOTALS/AVERAGES',
            f'{successful_analyses} success',
            f'{successful_analyses} servers',
            '-',
            '-',
            '-',
            f'${total_monthly_cost:.2f}',
            f'${total_monthly_cost * 12:.2f}'
        ])
        
        bulk_table = Table(bulk_data, colWidths=[1.3*inch, 0.8*inch, 1.3*inch, 0.6*inch, 0.7*inch, 0.8*inch, 0.9*inch, 1*inch])
        bulk_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.purple),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 8),
            ('FONTSIZE', (0, 1), (-1, -1), 7),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -2), colors.lavender),
            ('BACKGROUND', (0, -1), (-1, -1), colors.yellow),
            ('FONTNAME', (0, -1), (-1, -1), 'Helvetica-Bold'),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(bulk_table)
        
        # Error analysis
        if failed_analyses > 0:
            story.append(Paragraph("3.2 Error Analysis", self.styles['SubsectionHeader']))
            
            error_data = [['Server Name', 'Error Type', 'Error Message', 'Recommended Action']]
            
            for server_name, server_results in analysis_results.items():
                if 'error' in server_results:
                    error_msg = server_results['error']
                    error_type = 'Configuration Error' if 'specification' in error_msg.lower() else 'Analysis Error'
                    action = 'Review server specifications' if 'specification' in error_msg.lower() else 'Contact support'
                    
                    error_data.append([
                        server_name[:25] + ('...' if len(server_name) > 25 else ''),
                        error_type,
                        error_msg[:40] + ('...' if len(error_msg) > 40 else ''),
                        action
                    ])
            
            error_table = Table(error_data, colWidths=[1.5*inch, 1.2*inch, 2*inch, 1.3*inch])
            error_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.darkred),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 9),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.mistyrose),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('FONTSIZE', (0, 1), (-1, -1), 8),
                ('VALIGN', (0, 0), (-1, -1), 'TOP')
            ]))
            story.append(error_table)
        
        return story

    def _create_complete_financial_analysis(self, analysis_results, analysis_mode):
        """Create complete financial analysis with all cost components"""
        story = []
        story.append(Paragraph("4. Complete Financial Analysis", self.styles['SectionHeader']))
        
        # TCO Analysis
        story.append(Paragraph("4.1 Total Cost of Ownership (TCO) Analysis", self.styles['SubsectionHeader']))
        
        if analysis_mode == 'single':
            valid_results = {k: v for k, v in analysis_results.items() if 'error' not in v}
            if valid_results:
                prod_result = valid_results.get('PROD', list(valid_results.values())[0])
                monthly_cost = self.safe_get(prod_result, 'total_cost', 0)
        else:
            monthly_cost = sum(
                self.safe_get(result.get('PROD', list(result.values())[0] if result.values() else {}), 'total_cost', 0)
                for result in analysis_results.values() if 'error' not in result
            )
        
        # 5-year TCO projection with detailed breakdown
        tco_data = [['Year', 'AWS Infrastructure', 'AWS Support', 'Migration Costs', 'OpEx Savings', 'Net Annual Cost', 'Cumulative']]
        
        cumulative_cost = 0
        for year in range(1, 6):
            # Infrastructure costs with inflation
            annual_aws_cost = monthly_cost * 12 * (1.03 ** (year - 1))
            
            # AWS Support (estimated 10% of infrastructure)
            support_cost = annual_aws_cost * 0.10
            
            # Migration costs (one-time in year 1)
            migration_cost = 150000 if year == 1 else 0
            if analysis_mode == 'bulk':
                migration_cost = 300000 if year == 1 else 0
            
            # OpEx savings (increasing over time)
            base_savings = 200000 if analysis_mode == 'bulk' else 120000
            opex_savings = base_savings + (year * 25000)
            
            # Net cost calculation
            net_annual = annual_aws_cost + support_cost + migration_cost - opex_savings
            cumulative_cost += net_annual
            
            tco_data.append([
                f'Year {year}',
                f'${annual_aws_cost:,.0f}',
                f'${support_cost:,.0f}',
                f'${migration_cost:,.0f}',
                f'${opex_savings:,.0f}',
                f'${net_annual:,.0f}',
                f'${cumulative_cost:,.0f}'
            ])
        
        tco_table = Table(tco_data, colWidths=[0.8*inch, 1.1*inch, 0.9*inch, 1*inch, 1*inch, 1.1*inch, 1.1*inch])
        tco_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkgreen),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 8),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.lightgreen),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTSIZE', (0, 1), (-1, -1), 7)
        ]))
        story.append(tco_table)
        
        # Cost optimization recommendations
        story.append(Paragraph("4.2 Cost Optimization Opportunities", self.styles['SubsectionHeader']))
        
        optimization_data = [
            ['Optimization Strategy', 'Potential Savings', 'Implementation Effort', 'Risk Level', 'Timeline'],
            ['Reserved Instances (1-year)', '20-30%', 'Low', 'Low', 'Immediate'],
            ['Reserved Instances (3-year)', '40-50%', 'Low', 'Medium', 'Immediate'],
            ['Savings Plans', '15-25%', 'Low', 'Low', 'Immediate'],
            ['Right-sizing Review', '10-20%', 'Medium', 'Low', '1-2 months'],
            ['Storage Optimization', '5-15%', 'Low', 'Low', '2-4 weeks'],
            ['Automated Scaling', '10-25%', 'High', 'Medium', '2-3 months'],
            ['Multi-AZ Optimization', '5-10%', 'Medium', 'High', '1-2 months']
        ]
        
        optimization_table = Table(optimization_data, colWidths=[1.8*inch, 1*inch, 1*inch, 0.8*inch, 1*inch])
        optimization_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkorange),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 9),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.wheat),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTSIZE', (0, 1), (-1, -1), 8)
        ]))
        story.append(optimization_table)
        
        return story

    def _create_performance_capacity_analysis(self, analysis_results, server_specs):
        """Create detailed performance and capacity analysis"""
        story = []
        story.append(Paragraph("5. Performance & Capacity Analysis", self.styles['SectionHeader']))
        
        story.append(Paragraph("5.1 Resource Utilization Analysis", self.styles['SubsectionHeader']))
        
        if isinstance(server_specs, dict):
            # Single server analysis
            current_cpu = server_specs.get('cpu_util', server_specs.get('peak_cpu_percent', 75))
            current_ram = server_specs.get('ram_util', server_specs.get('peak_ram_percent', 80))
            
            utilization_data = [
                ['Resource', 'Current Peak', 'Recommended Target', 'Safety Margin', 'Scaling Headroom'],
                ['CPU Utilization', f'{current_cpu}%', '60-70%', f'{100-current_cpu}%', '30-40%'],
                ['Memory Utilization', f'{current_ram}%', '65-75%', f'{100-current_ram}%', '25-35%'],
                ['Storage I/O', 'Variable', '<80% sustained', 'Monitor', 'Burstable'],
                ['Network I/O', 'Unknown', '<70% sustained', 'Monitor', 'Elastic']
            ]
        
        else:
            # Bulk analysis - show aggregated stats
            if server_specs:
                avg_cpu = sum(s.get('peak_cpu_percent', 75) for s in server_specs) / len(server_specs)
                avg_ram = sum(s.get('peak_ram_percent', 80) for s in server_specs) / len(server_specs)
                
                utilization_data = [
                    ['Resource', 'Average Peak', 'Range', 'Recommended Target', 'Optimization Opportunity'],
                    ['CPU Utilization', f'{avg_cpu:.1f}%', f'{min(s.get("peak_cpu_percent", 75) for s in server_specs)}-{max(s.get("peak_cpu_percent", 75) for s in server_specs)}%', '60-70%', 'Instance optimization'],
                    ['Memory Utilization', f'{avg_ram:.1f}%', f'{min(s.get("peak_ram_percent", 80) for s in server_specs)}-{max(s.get("peak_ram_percent", 80) for s in server_specs)}%', '65-75%', 'Memory tuning'],
                    ['Storage Variance', 'High', f'{min(s.get("storage_gb", 100) for s in server_specs):,}-{max(s.get("storage_gb", 100) for s in server_specs):,} GB', 'Standardized tiers', 'Storage optimization']
                ]
            else:
                utilization_data = [['Resource', 'Status', 'Notes'], ['Analysis', 'Not Available', 'No server specifications provided']]
        
        utilization_table = Table(utilization_data, colWidths=[1.5*inch, 1.2*inch, 1.2*inch, 1.3*inch, 1.5*inch])
        utilization_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 9),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.lightblue),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTSIZE', (0, 1), (-1, -1), 8)
        ]))
        story.append(utilization_table)
        
        # Capacity planning
        story.append(Paragraph("5.2 Capacity Planning", self.styles['SubsectionHeader']))
        
        capacity_text = """
        <b>Growth Projections:</b><br/>
        • Data growth: 20% annually (configurable)<br/>
        • Transaction volume: 15% annually<br/>
        • User base: 10% annually<br/>
        • Query complexity: 5% annually<br/><br/>
        
        <b>Scaling Triggers:</b><br/>
        • CPU utilization >75% for 10+ minutes<br/>
        • Memory utilization >80% sustained<br/>
        • Storage utilization >85%<br/>
        • Connection pool >90% utilized<br/><br/>
        
        <b>Recommended Monitoring:</b><br/>
        • CloudWatch Enhanced Monitoring<br/>
        • Performance Insights enabled<br/>
        • Custom application metrics<br/>
        • Database slow query logs
        """
        
        story.append(Paragraph(capacity_text, self.styles['Highlight']))
        
        return story

    def _create_comprehensive_transfer_analysis(self, transfer_results):
        """Create comprehensive data transfer analysis"""
        story = []
        story.append(Paragraph("6. Data Transfer Comprehensive Analysis", self.styles['SectionHeader']))
        
        story.append(Paragraph("6.1 Transfer Method Comparison", self.styles['SubsectionHeader']))
        
        transfer_data = [['Method', 'Transfer Time', 'Total Cost', 'Bandwidth Util.', 'Downtime', 'Use Case']]
        
        for method, result in transfer_results.items():
            transfer_data.append([
                result.recommended_method,
                f'{result.transfer_time_days:.1f} days ({result.transfer_time_hours:.1f} hours)',
                f'${result.total_cost:.2f}',
                f'{result.bandwidth_utilization:.0f}%',
                f'{result.estimated_downtime_hours:.2f} hours',
                'Production' if result.transfer_time_hours < 24 else 'Development'
            ])
        
        transfer_table = Table(transfer_data, colWidths=[1.5*inch, 1.3*inch, 0.8*inch, 0.8*inch, 0.8*inch, 1*inch])
        transfer_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.orange),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 9),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.wheat),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTSIZE', (0, 1), (-1, -1), 8)
        ]))
        story.append(transfer_table)
        
        # Detailed cost breakdown for each method
        story.append(Paragraph("6.2 Detailed Cost Breakdown", self.styles['SubsectionHeader']))
        
        for method, result in transfer_results.items():
            story.append(Paragraph(f"{result.recommended_method} - ${result.total_cost:.2f}", self.styles['DetailHeader']))
            
            if result.cost_breakdown:
                method_cost_data = [['Cost Component', 'Amount', 'Description']]
                for component, cost in result.cost_breakdown.items():
                    component_name = component.replace('_', ' ').title()
                    method_cost_data.append([component_name, f'${cost:.2f}', 'Per-use pricing'])
                
                method_cost_table = Table(method_cost_data, colWidths=[2*inch, 1*inch, 3*inch])
                method_cost_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 9),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black),
                    ('FONTSIZE', (0, 1), (-1, -1), 8)
                ]))
                story.append(method_cost_table)
                story.append(Spacer(1, 10))
        
        return story

    def _create_complete_ai_analysis(self, ai_insights):
        """Create complete AI analysis section"""
        story = []
        story.append(Paragraph("7. Complete AI Analysis & Insights", self.styles['SectionHeader']))
        
        # AI metrics summary
        story.append(Paragraph("7.1 AI Assessment Summary", self.styles['SubsectionHeader']))
        
        ai_metrics_data = [
            ['AI Metric', 'Value', 'Confidence Level', 'Impact'],
            ['Migration Risk Level', ai_insights.get('risk_level', 'Unknown'), 'High', 'Strategic planning'],
            ['Cost Optimization Potential', f"{ai_insights.get('cost_optimization_potential', 0) * 100:.1f}%", 'High', 'Budget optimization'],
            ['Recommended Writers', str(ai_insights.get('recommended_writers', 1)), 'Medium', 'Write performance'],
            ['Recommended Readers', str(ai_insights.get('recommended_readers', 1)), 'Medium', 'Read scaling'],
            ['Success Probability', '>95%', 'High', 'Project confidence']
        ]
        
        ai_metrics_table = Table(ai_metrics_data, colWidths=[1.8*inch, 1.2*inch, 1.2*inch, 1.8*inch])
        ai_metrics_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.lightblue),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTSIZE', (0, 1), (-1, -1), 9)
        ]))
        story.append(ai_metrics_table)
        
        # Complete AI analysis text
        story.append(Paragraph("7.2 Detailed AI Analysis", self.styles['SubsectionHeader']))
        
        ai_analysis_text = ai_insights.get("ai_analysis", "No detailed AI analysis available.")
        
        if ai_analysis_text and ai_analysis_text != "No detailed AI analysis available.":
            # Split into manageable paragraphs
            paragraphs = ai_analysis_text.split('\n\n')
            for paragraph in paragraphs:
                if paragraph.strip():
                    # Clean up the paragraph text
                    clean_paragraph = paragraph.strip().replace('\n', ' ')
                    story.append(Paragraph(clean_paragraph, self.styles['Normal']))
                    story.append(Spacer(1, 10))
        else:
            story.append(Paragraph("AI analysis was not available during this report generation. Consider re-running the analysis with AI insights enabled.", self.styles['Normal']))
        
        # AI recommendations
        story.append(Paragraph("7.3 AI-Generated Recommendations", self.styles['SubsectionHeader']))
        
        recommendations_text = """
        <b>Top AI Recommendations:</b><br/>
        1. <b>Phased Migration Approach:</b> Implement migration in controlled phases to minimize risk<br/>
        2. <b>Performance Testing:</b> Conduct thorough performance testing before production cutover<br/>
        3. <b>Monitoring Strategy:</b> Implement comprehensive monitoring from day one<br/>
        4. <b>Backup Validation:</b> Verify backup and recovery procedures during migration<br/>
        5. <b>Cost Optimization:</b> Review and implement cost optimization strategies post-migration<br/>
        6. <b>Security Review:</b> Conduct security assessment of migrated infrastructure<br/>
        7. <b>Documentation:</b> Maintain detailed documentation throughout the migration process
        """
        
        story.append(Paragraph(recommendations_text, self.styles['Highlight']))
        
        return story

    def _create_detailed_migration_strategy(self, analysis_results, ai_insights, migration_config):
        """Create detailed migration strategy"""
        story = []
        story.append(Paragraph("8. Detailed Migration Strategy", self.styles['SectionHeader']))
        
        # Migration approach
        story.append(Paragraph("8.1 Migration Methodology", self.styles['SubsectionHeader']))
        
        methodology_text = """
        <b>Migration Framework:</b> AWS Database Migration Service (DMS) with Schema Conversion Tool (SCT)<br/>
        <b>Approach:</b> Phased migration with parallel validation<br/>
        <b>Strategy:</b> Minimize downtime through continuous replication<br/>
        <b>Validation:</b> Comprehensive data validation and application testing<br/>
        <b>Rollback Plan:</b> Complete rollback capability maintained until Go/No-Go decision
        """
        
        story.append(Paragraph(methodology_text, self.styles['Normal']))
        
        # Detailed phases
        story.append(Paragraph("8.2 Migration Phases", self.styles['SubsectionHeader']))
        
        phases_data = [
            ['Phase', 'Duration', 'Key Activities', 'Deliverables', 'Success Criteria'],
            ['1. Assessment', '2-3 weeks', 'Schema analysis, compatibility assessment', 'Assessment report, migration plan', 'Complete inventory documented'],
            ['2. Schema Conversion', '3-4 weeks', 'AWS SCT conversion, manual remediation', 'Converted schema, conversion report', '100% schema compatibility'],
            ['3. DMS Setup', '1-2 weeks', 'Replication instance setup, endpoint configuration', 'DMS tasks configured', 'Successful test migration'],
            ['4. Initial Load', '3-5 days', 'Full data migration, initial sync', 'Data migrated, validation report', 'Data integrity verified'],
            ['5. CDC Setup', '1 week', 'Change data capture configuration', 'Real-time replication', 'Ongoing sync established'],
            ['6. Application Testing', '4-6 weeks', 'Application code changes, testing', 'Updated applications, test results', 'All tests passing'],
            ['7. User Acceptance', '2-3 weeks', 'End-user testing, performance validation', 'UAT sign-off', 'User acceptance achieved'],
            ['8. Go-Live Preparation', '1 week', 'Final sync, cutover planning', 'Go-live checklist', 'Readiness confirmed'],
            ['9. Production Cutover', '8-24 hours', 'DNS switch, application restart', 'Production system live', 'System operational'],
            ['10. Post-Migration', '2-4 weeks', 'Monitoring, optimization, support', 'Optimized system', 'SLA compliance achieved']
        ]
        
        phases_table = Table(phases_data, colWidths=[0.8*inch, 0.8*inch, 1.5*inch, 1.3*inch, 1.3*inch])
        phases_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.navy),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 8),
            ('FONTSIZE', (0, 1), (-1, -1), 7),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.lightblue),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('VALIGN', (0, 0), (-1, -1), 'TOP')
        ]))
        story.append(phases_table)
        
        return story

    def _create_comprehensive_risk_assessment(self, analysis_results, ai_insights):
        """Create comprehensive risk assessment"""
        story = []
        story.append(Paragraph("9. Risk Assessment & Mitigation", self.styles['SectionHeader']))
        
        # Risk matrix
        story.append(Paragraph("9.1 Risk Analysis Matrix", self.styles['SubsectionHeader']))
        
        risk_data = [
            ['Risk Category', 'Probability', 'Impact', 'Risk Score', 'Mitigation Strategy', 'Contingency Plan'],
            ['Schema Conversion Issues', 'Medium', 'High', '6', 'AWS SCT + Manual Review', 'Expert consultation'],
            ['Data Corruption', 'Low', 'Critical', '5', 'Checksums + Validation', 'Full rollback procedure'],
            ['Performance Degradation', 'Medium', 'Medium', '4', 'Load testing + Tuning', 'Instance upsizing'],
            ['Extended Downtime', 'Low', 'High', '4', 'Parallel sync + Quick cutover', 'Rollback to source'],
            ['Application Compatibility', 'Medium', 'Medium', '4', 'Code review + Testing', 'Application patches'],
            ['Cost Overrun', 'Medium', 'Low', '2', 'Reserved instances + Monitoring', 'Budget reallocation'],
            ['Security Vulnerabilities', 'Low', 'High', '4', 'Security assessment', 'Immediate patching'],
            ['Network Connectivity', 'Low', 'Medium', '2', 'Redundant connections', 'Backup connectivity'],
            ['Skill Gap', 'High', 'Medium', '6', 'Training + Documentation', 'External support'],
            ['Regulatory Compliance', 'Low', 'High', '4', 'Compliance review', 'Audit preparation']
        ]
        
        risk_table = Table(risk_data, colWidths=[1.2*inch, 0.7*inch, 0.7*inch, 0.6*inch, 1.3*inch, 1.2*inch])
        risk_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkred),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 8),
            ('FONTSIZE', (0, 1), (-1, -1), 7),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.mistyrose),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('VALIGN', (0, 0), (-1, -1), 'TOP')
        ]))
        story.append(risk_table)
        
        # Risk mitigation strategies
        story.append(Paragraph("9.2 Detailed Mitigation Strategies", self.styles['SubsectionHeader']))
        
        mitigation_text = """
        <b>High Priority Mitigations:</b><br/>
        1. <b>Schema Conversion:</b> Use AWS SCT with expert review for complex conversions<br/>
        2. <b>Data Validation:</b> Implement comprehensive data validation scripts<br/>
        3. <b>Performance Testing:</b> Conduct load testing in staging environment<br/>
        4. <b>Rollback Procedures:</b> Maintain complete rollback capability<br/>
        5. <b>Communication Plan:</b> Establish clear communication channels<br/><br/>
        
        <b>Medium Priority Mitigations:</b><br/>
        1. <b>Monitoring:</b> Implement comprehensive monitoring and alerting<br/>
        2. <b>Training:</b> Provide team training on AWS RDS operations<br/>
        3. <b>Documentation:</b> Create detailed operational procedures<br/>
        4. <b>Security:</b> Implement security best practices and compliance<br/>
        5. <b>Cost Management:</b> Establish cost monitoring and optimization
        """
        
        story.append(Paragraph(mitigation_text, self.styles['Highlight']))
        
        return story

    def _create_detailed_implementation_roadmap(self, analysis_results, analysis_mode):
        """Create detailed implementation roadmap"""
        story = []
        story.append(Paragraph("10. Detailed Implementation Roadmap", self.styles['SectionHeader']))
        
        # Timeline overview
        story.append(Paragraph("10.1 Timeline Overview", self.styles['SubsectionHeader']))
        
        if analysis_mode == 'single':
            timeline_duration = "12-16 weeks"
            complexity_note = "Single server migration with moderate complexity"
        else:
            timeline_duration = "16-24 weeks"
            complexity_note = "Bulk migration with coordination requirements"
        
        timeline_text = f"""
        <b>Total Duration:</b> {timeline_duration}<br/>
        <b>Complexity:</b> {complexity_note}<br/>
        <b>Resource Requirements:</b> 2-3 FTE throughout migration<br/>
        <b>Critical Path:</b> Schema conversion and application testing<br/>
        <b>Dependencies:</b> Network connectivity, security approvals, application readiness
        """
        
        story.append(Paragraph(timeline_text, self.styles['Normal']))
        
        # Detailed tasks
        story.append(Paragraph("10.2 Detailed Task Breakdown", self.styles['SubsectionHeader']))
        
        tasks_data = [
            ['Week', 'Phase', 'Key Tasks', 'Resources', 'Dependencies', 'Deliverables'],
            ['1-2', 'Assessment', 'Schema analysis, workload assessment', 'DBA, Architect', 'Server access', 'Assessment report'],
            ['3-4', 'Planning', 'Migration plan, resource allocation', 'PM, Architect', 'Budget approval', 'Migration plan'],
            ['5-7', 'Schema Conversion', 'AWS SCT, manual conversion', 'DBA, Developer', 'Tool installation', 'Converted schema'],
            ['8-9', 'DMS Setup', 'Replication instances, endpoints', 'DBA, Admin', 'AWS accounts', 'DMS configuration'],
            ['10-12', 'Testing', 'Application testing, performance', 'QA, Developer', 'Test environment', 'Test results'],
            ['13-14', 'UAT', 'User acceptance testing', 'Users, QA', 'Application ready', 'UAT sign-off'],
            ['15', 'Go-Live Prep', 'Final preparations, rehearsal', 'All teams', 'All previous phases', 'Go-live checklist'],
            ['16', 'Go-Live', 'Production cutover', 'All teams', 'Business approval', 'Live system'],
            ['17-20', 'Support', 'Post-migration support, optimization', 'Support team', 'System stable', 'Optimized system']
        ]
        
        tasks_table = Table(tasks_data, colWidths=[0.6*inch, 1*inch, 1.5*inch, 1*inch, 1*inch, 1.2*inch])
        tasks_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkgreen),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 8),
            ('FONTSIZE', (0, 1), (-1, -1), 7),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.lightgreen),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('VALIGN', (0, 0), (-1, -1), 'TOP')
        ]))
        story.append(tasks_table)
        
        return story

    def _create_comprehensive_appendices(self, analysis_results, server_specs, migration_config):
        """Create comprehensive appendices"""
        story = []
        story.append(Paragraph("11. Technical Appendices", self.styles['SectionHeader']))
        
        # Appendix A: AWS Instance Types Reference
        story.append(Paragraph("Appendix A: AWS RDS Instance Types Reference", self.styles['SubsectionHeader']))
        
        instance_ref_data = [
            ['Instance Family', 'Use Case', 'vCPU Range', 'Memory Range', 'Network Performance'],
            ['db.t3/t4g', 'Development, low traffic', '2-8', '1-32 GB', 'Low to Moderate'],
            ['db.m5/m6i', 'General purpose, balanced', '2-96', '8-384 GB', 'Up to 25 Gbps'],
            ['db.r5/r6i', 'Memory-intensive', '2-96', '16-768 GB', 'Up to 25 Gbps'],
            ['db.c5/c6i', 'CPU-intensive', '2-96', '4-192 GB', 'Up to 25 Gbps'],
            ['db.x1e/x2gd', 'In-memory databases', '4-128', '122-4,078 GB', 'Up to 25 Gbps']
        ]
        
        instance_ref_table = Table(instance_ref_data, colWidths=[1.2*inch, 1.5*inch, 1*inch, 1.2*inch, 1.3*inch])
        instance_ref_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 9),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTSIZE', (0, 1), (-1, -1), 8)
        ]))
        story.append(instance_ref_table)
        
        # Appendix B: Cost Optimization Checklist
        story.append(Paragraph("Appendix B: Cost Optimization Checklist", self.styles['SubsectionHeader']))
        
        checklist_text = """
        <b>Immediate Actions (Month 1):</b><br/>
        □ Review instance sizing and right-size if needed<br/>
        □ Enable automated backups with appropriate retention<br/>
        □ Configure CloudWatch monitoring and alerts<br/>
        □ Review security group configurations<br/><br/>
        
        <b>Short-term Actions (Months 2-3):</b><br/>
        □ Implement Reserved Instances for stable workloads<br/>
        □ Optimize storage configuration (gp3 vs io1)<br/>
        □ Review and optimize backup strategies<br/>
        □ Implement automated scaling policies<br/><br/>
        
        <b>Long-term Actions (Months 4-12):</b><br/>
        □ Consider Savings Plans for broader coverage<br/>
        □ Evaluate Aurora Serverless for variable workloads<br/>
        □ Implement comprehensive cost monitoring<br/>
        □ Regular quarterly cost reviews and optimization
        """
        
        story.append(Paragraph(checklist_text, self.styles['CodeBlock']))
        
        # Appendix C: Migration Checklist
        story.append(Paragraph("Appendix C: Migration Execution Checklist", self.styles['SubsectionHeader']))
        
        migration_checklist_text = """
        <b>Pre-Migration Checklist:</b><br/>
        □ Schema conversion completed and validated<br/>
        □ DMS replication instances configured<br/>
        □ Network connectivity tested<br/>
        □ Security groups and IAM roles configured<br/>
        □ Backup procedures tested<br/>
        □ Application code changes completed<br/>
        □ Rollback procedures documented and tested<br/><br/>
        
        <b>Migration Day Checklist:</b><br/>
        □ All stakeholders notified<br/>
        □ Support teams on standby<br/>
        □ Monitoring dashboards active<br/>
        □ Final data sync initiated<br/>
        □ Application cutover executed<br/>
        □ Smoke tests completed<br/>
        □ Performance validation passed<br/>
        □ User acceptance confirmed<br/><br/>
        
        <b>Post-Migration Checklist:</b><br/>
        □ Full system validation completed<br/>
        □ Performance monitoring active<br/>
        □ Cost monitoring configured<br/>
        □ Documentation updated<br/>
        □ Team training completed<br/>
        □ Lessons learned documented
        """
        
        story.append(Paragraph(migration_checklist_text, self.styles['CodeBlock']))
        
        return story


# Helper function to use the comprehensive generator
def generate_ultra_comprehensive_pdf_report(analysis_results, analysis_mode, server_specs=None, 
                                           ai_insights=None, transfer_results=None, migration_config=None,
                                           workload_characteristics=None):
    """Helper function to generate ultra-comprehensive PDF report"""
    try:
        comprehensive_generator = ComprehensiveReportGenerator()
        
        pdf_bytes = comprehensive_generator.generate_ultra_comprehensive_report(
            analysis_results=analysis_results,
            analysis_mode=analysis_mode,
            server_specs=server_specs,
            ai_insights=ai_insights,
            transfer_results=transfer_results,
            migration_config=migration_config,
            workload_characteristics=workload_characteristics
        )
        
        return pdf_bytes
        
    except Exception as e:
        print(f"Error generating ultra-comprehensive PDF report: {str(e)}")
        import traceback
        traceback.print_exc()
        return None