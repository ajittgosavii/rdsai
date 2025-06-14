import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import time
import asyncio
import traceback
import json
import base64
from datetime import datetime, timedelta
import os
from typing import Dict, Any, Optional
from io import BytesIO
import tempfile
import firebase_admin
from firebase_admin import credentials, auth, firestore
from data_transfer_calculator import DataTransferCalculator, TransferMethod, TransferMethodResult
import io
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
import plotly.io as pio
import kaleido  # Required for plotly image export


# Authentication imports
import bcrypt
import jwt

# Import our enhanced modules
try:
    from rds_sizing import EnhancedRDSSizingCalculator, MigrationType, WorkloadCharacteristics
    from aws_pricing import EnhancedAWSPricingAPI
except ImportError as e:
    st.error(f"Import error: {e}")
    st.stop()

# Configure Streamlit
st.set_page_config(
    page_title="Enterprise AWS RDS Migration & Sizing Tool",
    layout="wide",
    page_icon="🚀",
    initial_sidebar_state="expanded"
)

# ================================
# AUTHENTICATION FUNCTIONS
# ================================

def hash_password(password: str) -> str:
    """Hash a password for storing."""
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
    return hashed.decode('utf-8')

def verify_password(password: str, hashed: str) -> bool:
    """Verify a stored password against one provided by user"""
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

def get_users_from_secrets():
    """Get users from Streamlit secrets"""
    try:
        if "auth" in st.secrets and "users" in st.secrets["auth"]:
            return dict(st.secrets["auth"]["users"])
        return {}
    except Exception as e:
        st.error(f"Error loading users: {e}")
        return {}

def authenticate_user(email: str, password: str) -> dict:
    """Authenticate user credentials"""
    users = get_users_from_secrets()
    
    for username, user_data in users.items():
        user_dict = dict(user_data)
        if user_dict.get('email', '').lower() == email.lower():
            if verify_password(password, user_dict['password']):
                return {
                    'username': username,
                    'email': user_dict['email'],
                    'name': user_dict['name'],
                    'role': user_dict['role'],
                    'authenticated': True
                }
    return {'authenticated': False}

def create_session_token(user_data: dict) -> str:
    """Create a JWT session token"""
    try:
        config = dict(st.secrets["auth"]["config"])
        payload = {
            'username': user_data['username'],
            'email': user_data['email'],
            'name': user_data['name'],
            'role': user_data['role'],
            'exp': datetime.utcnow() + timedelta(days=int(config.get('cookie_expiry_days', 7)))
        }
        return jwt.encode(payload, config['cookie_key'], algorithm='HS256')
    except Exception as e:
        st.error(f"Error creating session token: {e}")
        return None

def verify_session_token(token: str) -> dict:
    """Verify and decode session token"""
    try:
        config = dict(st.secrets["auth"]["config"])
        payload = jwt.decode(token, config['cookie_key'], algorithms=['HS256'])
        return {
            'username': payload['username'],
            'email': payload['email'],
            'name': payload['name'],
            'role': payload['role'],
            'authenticated': True
        }
    except jwt.ExpiredSignatureError:
        return {'authenticated': False, 'error': 'Session expired'}
    except jwt.InvalidTokenError:
        return {'authenticated': False, 'error': 'Invalid session'}

def show_login_form():
    """Display the login form"""
    st.markdown("""
    <div style="max-width: 400px; margin: 50px auto; padding: 30px; 
                border: 1px solid #ddd; border-radius: 10px; 
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">
    """, unsafe_allow_html=True)
    
    st.markdown("### 🔐 Enterprise RDS Migration Tool - Login")
    st.markdown("Please enter your credentials to access the system.")
    
    with st.form("login_form"):
        email = st.text_input("📧 Email", placeholder="user@yourcompany.com")
        password = st.text_input("🔒 Password", type="password", placeholder="Enter your password")
        
        col1, col2 = st.columns([1, 1])
        with col1:
            login_button = st.form_submit_button("🚀 Login", use_container_width=True)
        with col2:
            if st.form_submit_button("👥 Show Test Users", use_container_width=True):
                st.session_state.show_test_users = True
    
    if login_button:
        if email and password:
            with st.spinner("Authenticating..."):
                user_data = authenticate_user(email, password)
                
                if user_data['authenticated']:
                    token = create_session_token(user_data)
                    if token:
                        st.session_state.user_authenticated = True
                        st.session_state.user_data = user_data
                        st.session_state.session_token = token
                        st.session_state.user_id = user_data['username']
                        st.session_state.is_logged_in = True
                        
                        st.success(f"✅ Welcome, {user_data['name']}!")
                        st.rerun()
                    else:
                        st.error("❌ Error creating session. Please try again.")
                else:
                    st.error("❌ Invalid email or password. Please try again.")
        else:
            st.error("❌ Please enter both email and password.")
    
    if st.session_state.get('show_test_users', False):
        st.markdown("---")
        st.markdown("#### 👥 Test Users (Development Only)")
        users = get_users_from_secrets()
        for username, user_data in users.items():
            user_dict = dict(user_data)
            st.markdown(f"**{user_dict['name']}** ({user_dict['role']})")
            st.code(f"Email: {user_dict['email']}")
        st.markdown("*Passwords are set in secrets.toml*")
    
    st.markdown("</div>", unsafe_allow_html=True)

def logout_user():
    """Logout the current user"""
    auth_keys = [
        'user_authenticated', 'user_data', 'session_token', 
        'user_id', 'user_email', 'is_logged_in'
    ]
    for key in auth_keys:
        if key in st.session_state:
            del st.session_state[key]
    
    st.success("👋 You have been logged out successfully!")
    st.rerun()

def check_authentication():
    """Check if user is authenticated and handle session"""
    if 'user_authenticated' not in st.session_state:
        st.session_state.user_authenticated = False
    
    if not st.session_state.user_authenticated and 'session_token' in st.session_state:
        token_data = verify_session_token(st.session_state.session_token)
        if token_data['authenticated']:
            st.session_state.user_authenticated = True
            st.session_state.user_data = token_data
            st.session_state.user_id = token_data['username']
            st.session_state.is_logged_in = True
        else:
            if 'session_token' in st.session_state:
                del st.session_state['session_token']
    
    return st.session_state.user_authenticated

def show_user_info():
    """Display current user info in sidebar"""
    if st.session_state.get('user_authenticated', False) and 'user_data' in st.session_state:
        user_data = st.session_state.user_data
        
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 👤 Current User")
        st.sidebar.markdown(f"**Name:** {user_data['name']}")
        st.sidebar.markdown(f"**Email:** {user_data['email']}")
        st.sidebar.markdown(f"**Role:** {user_data['role'].title()}")
        
        if st.sidebar.button("🚪 Logout", use_container_width=True):
            logout_user()

# ================================
# FIREBASE FUNCTIONS
# ================================

@st.cache_resource(ttl=3600)
def initialize_firebase():
    """Initializes the Firebase app and authenticates the user."""
    try:
        if firebase_admin._apps:
            st.info("Firebase already initialized")
            app = firebase_admin.get_app()
            return app, auth, firestore.client()
        
        if "connections" not in st.secrets or "firebase" not in st.secrets["connections"]:
            st.error("Firebase configuration not found in Streamlit secrets.")
            return None, None, None

        firebase_config_dict = dict(st.secrets["connections"]["firebase"])
        
        required_fields = ['project_id', 'private_key', 'client_email']
        missing_fields = [field for field in required_fields if not firebase_config_dict.get(field)]
        
        if missing_fields:
            st.error(f"Missing required Firebase fields: {missing_fields}")
            return None, None, None

        firebase_config_dict['type'] = 'service_account'
        
        private_key = firebase_config_dict.get('private_key', '').strip()
        
        if not private_key.startswith('-----BEGIN PRIVATE KEY-----'):
            st.error("Private key missing BEGIN header")
            return None, None, None
        
        if not private_key.endswith('-----END PRIVATE KEY-----'):
            st.error("Private key missing END footer")
            return None, None, None

        cred = credentials.Certificate(firebase_config_dict)
        firebase_app = firebase_admin.initialize_app(
            cred, 
            options={'projectId': firebase_config_dict['project_id']}
        )
        
        db_client = firestore.client(firebase_app)
        
        st.success("🎉 Firebase Admin SDK initialized successfully!")
        
        return firebase_app, auth, db_client
        
    except Exception as e:
        st.error(f"Firebase initialization failed: {e}")
        import traceback
        st.code(traceback.format_exc())
        return None, None, None

# ================================
# UTILITY FUNCTIONS
# ================================

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

# ================================
# ENHANCED REPORT GENERATOR
# ================================

# Enhanced PDF Report Generator with Charts and Detailed Analysis
# This code replaces the existing EnhancedReportGenerator class in your streamlit_app.py

import io
import base64
import tempfile
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.charts.piecharts import Pie
from reportlab.graphics.charts.linecharts import HorizontalLineChart
from reportlab.lib.colors import HexColor

# COMPLETE FIX: Replace the entire setup_custom_styles method
# Find this method in your ComprehensiveReportGenerator class (around line 332) and replace it:
# ================================
# FIXED COMPREHENSIVE REPORT GENERATOR CLASS
# ================================

# ================================
# IMPROVED COMPREHENSIVE REPORT GENERATOR CLASS
# ================================

import io
import base64
import tempfile
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT

class ImprovedReportGenerator:
    """Enhanced PDF Report Generator with Better Formatting and Layout"""
    
    def __init__(self):
        """Initialize the improved report generator"""
        self.styles = getSampleStyleSheet()
        self.chart_width = 5.5*inch
        self.chart_height = 3.5*inch
        self.setup_custom_styles()
        
        # Debug: Verify styles are created
        self._verify_styles()
    
    def _verify_styles(self):
        """Verify that all required styles are available"""
        required_styles = [
            'ReportTitle', 'SectionHeader', 'SubsectionHeader', 
            'BodyText', 'KeyMetric', 'TableHeader', 'BulletPoint'
        ]
        
        for style_name in required_styles:
            if style_name not in self.styles.byName:
                print(f"Warning: Style '{style_name}' not found. Creating fallback.")
                self._create_fallback_style(style_name)
    
    def _create_fallback_style(self, style_name):
        """Create a fallback style if the original creation failed"""
        try:
            base_style = self.styles['Normal']
            if style_name == 'ReportTitle':
                self.styles.add(ParagraphStyle(
                    name='ReportTitle',
                    parent=self.styles['Title'],
                    fontSize=24,
                    fontName='Helvetica-Bold',
                    alignment=TA_CENTER
                ))
            elif style_name == 'SectionHeader':
                self.styles.add(ParagraphStyle(
                    name='SectionHeader',
                    parent=self.styles['Heading1'],
                    fontSize=16,
                    fontName='Helvetica-Bold',
                    spaceBefore=20,
                    spaceAfter=12
                ))
            else:
                # Generic fallback
                self.styles.add(ParagraphStyle(
                    name=style_name,
                    parent=base_style,
                    fontSize=12,
                    fontName='Helvetica'
                ))
        except Exception as e:
            print(f"Error creating fallback style {style_name}: {e}")
    
    def setup_custom_styles(self):
        """Setup improved custom styles for the report"""
        try:
            # Report Title Style
            self.styles.add(ParagraphStyle(
                name='ReportTitle',
                parent=self.styles['Title'],
                fontSize=24,
                spaceBefore=0,
                spaceAfter=20,
                textColor=colors.darkblue,
                alignment=TA_CENTER,
                fontName='Helvetica-Bold'
            ))
            
            # Section Header Style
            self.styles.add(ParagraphStyle(
                name='SectionHeader',
                parent=self.styles['Heading1'],
                fontSize=16,
                spaceBefore=20,
                spaceAfter=12,
                textColor=colors.darkblue,
                fontName='Helvetica-Bold',
                borderWidth=1,
                borderColor=colors.darkblue,
                borderPadding=5
            ))
            
            # Subsection Header Style
            self.styles.add(ParagraphStyle(
                name='SubsectionHeader',
                parent=self.styles['Heading2'],
                fontSize=14,
                spaceBefore=15,
                spaceAfter=8,
                textColor=colors.darkgreen,
                fontName='Helvetica-Bold'
            ))
            
            # Body Text Style
            self.styles.add(ParagraphStyle(
                name='BodyText',
                parent=self.styles['Normal'],
                fontSize=11,
                spaceBefore=6,
                spaceAfter=6,
                textColor=colors.black,
                fontName='Helvetica',
                leading=14
            ))
            
            # Key Metric Style
            self.styles.add(ParagraphStyle(
                name='KeyMetric',
                parent=self.styles['Normal'],
                fontSize=12,
                textColor=colors.darkred,
                fontName='Helvetica-Bold',
                alignment=TA_CENTER
            ))
            
            # Table Header Style
            self.styles.add(ParagraphStyle(
                name='TableHeader',
                parent=self.styles['Normal'],
                fontSize=10,
                textColor=colors.white,
                fontName='Helvetica-Bold',
                alignment=TA_CENTER
            ))
            
            # Bullet Point Style
            self.styles.add(ParagraphStyle(
                name='BulletPoint',
                parent=self.styles['Normal'],
                fontSize=11,
                spaceBefore=4,
                spaceAfter=4,
                leftIndent=20,
                bulletIndent=10,
                fontName='Helvetica'
            ))
            
        except Exception as e:
            print(f"Error setting up custom styles: {e}")
            self._create_basic_styles()
    
    def _create_basic_styles(self):
        """Create basic styles as fallback"""
        basic_styles = {
            'ReportTitle': ('Title', 24),
            'SectionHeader': ('Heading1', 16),
            'SubsectionHeader': ('Heading2', 14),
            'BodyText': ('Normal', 11),
            'KeyMetric': ('Normal', 12),
            'TableHeader': ('Normal', 10),
            'BulletPoint': ('Normal', 11)
        }
        
        for style_name, (parent, size) in basic_styles.items():
            try:
                if style_name not in self.styles.byName:
                    self.styles.add(ParagraphStyle(
                        name=style_name,
                        parent=self.styles[parent],
                        fontSize=size,
                        fontName='Helvetica'
                    ))
            except Exception as e:
                print(f"Error creating basic style {style_name}: {e}")
    
    def safe_get_style(self, style_name, fallback='Normal'):
        """Safely get a style, falling back to Normal if not found"""
        try:
            return self.styles[style_name]
        except KeyError:
            print(f"Style '{style_name}' not found, using '{fallback}'")
            return self.styles[fallback]
    
    def create_improved_cost_chart(self, analysis_results, analysis_mode):
        """Create an improved cost breakdown chart with better formatting"""
        try:
            if analysis_mode == 'single':
                valid_results = {k: v for k, v in analysis_results.items() if 'error' not in v}
                if not valid_results:
                    return None
                
                prod_result = valid_results.get('PROD', list(valid_results.values())[0])
                
                # Extract cost components with better logic
                cost_data = {}
                
                if 'writer' in prod_result:
                    # Aurora/RDS Cluster setup
                    writer_cost = prod_result.get('cost_breakdown', {}).get('writer_monthly', 0)
                    readers_cost = prod_result.get('cost_breakdown', {}).get('readers_monthly', 0)
                    storage_cost = prod_result.get('cost_breakdown', {}).get('storage_monthly', 0)
                    backup_cost = prod_result.get('cost_breakdown', {}).get('backup_monthly', 0)
                    transfer_cost = prod_result.get('cost_breakdown', {}).get('transfer_monthly', 50)
                    
                    if writer_cost > 0:
                        cost_data['Writer Instance'] = writer_cost
                    if readers_cost > 0:
                        cost_data['Reader Instances'] = readers_cost
                    if storage_cost > 0:
                        cost_data['Storage'] = storage_cost
                    if backup_cost > 0:
                        cost_data['Backup'] = backup_cost
                    if transfer_cost > 0:
                        cost_data['Data Transfer'] = transfer_cost
                else:
                    # Standard RDS setup
                    cost_breakdown = prod_result.get('cost_breakdown', {})
                    instance_cost = cost_breakdown.get('instance_monthly', prod_result.get('instance_cost', 0))
                    storage_cost = cost_breakdown.get('storage_monthly', prod_result.get('storage_cost', 0))
                    backup_cost = cost_breakdown.get('backup_monthly', storage_cost * 0.1)
                    transfer_cost = cost_breakdown.get('transfer_monthly', 50)
                    
                    if instance_cost > 0:
                        cost_data['Compute Instance'] = instance_cost
                    if storage_cost > 0:
                        cost_data['Storage'] = storage_cost
                    if backup_cost > 0:
                        cost_data['Backup'] = backup_cost
                    if transfer_cost > 0:
                        cost_data['Data Transfer'] = transfer_cost
                
                # Filter out zero values and ensure we have data
                cost_data = {k: v for k, v in cost_data.items() if v > 0}
                
                if not cost_data:
                    # Fallback data if no breakdown available
                    total_cost = prod_result.get('total_cost', 0)
                    cost_data = {
                        'Database Instance': total_cost * 0.7,
                        'Storage & Backup': total_cost * 0.25,
                        'Data Transfer': total_cost * 0.05
                    }
                
                # Create improved pie chart
                colors_list = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
                
                fig = go.Figure(data=[go.Pie(
                    labels=list(cost_data.keys()),
                    values=list(cost_data.values()),
                    hole=.4,
                    textinfo='label+percent',
                    textposition='outside',
                    texttemplate='%{label}<br>%{percent}<br>$%{value:,.0f}',
                    textfont={"size": 10},
                    marker=dict(colors=colors_list[:len(cost_data)], line=dict(color='#FFFFFF', width=2))
                )])
                
                fig.update_layout(
                    title={
                        'text': f'Monthly Cost Breakdown - ${sum(cost_data.values()):,.0f}',
                        'x': 0.5,
                        'font': {'size': 14, 'color': 'darkblue'},
                        'y': 0.95
                    },
                    font=dict(size=10),
                    showlegend=True,
                    legend=dict(
                        orientation="v", 
                        yanchor="middle", 
                        y=0.5, 
                        xanchor="left", 
                        x=1.05,
                        font=dict(size=9)
                    ),
                    margin=dict(t=50, b=50, l=50, r=100),
                    width=600,
                    height=400
                )
                
                return self.create_plotly_chart_image(fig, width=600, height=400)
            
            else:  # Bulk analysis
                server_costs = []
                server_names = []
                
                for server_name, server_results in analysis_results.items():
                    if 'error' not in server_results:
                        result = server_results.get('PROD', list(server_results.values())[0])
                        if 'error' not in result:
                            cost = result.get('total_cost', 0)
                            if cost > 0:
                                server_costs.append(cost)
                                # Truncate long server names for better display
                                display_name = server_name[:15] + '...' if len(server_name) > 15 else server_name
                                server_names.append(display_name)
                
                if not server_costs:
                    return None
                
                # Create improved bar chart for bulk analysis
                fig = go.Figure(data=[go.Bar(
                    x=server_names,
                    y=server_costs,
                    marker_color='lightblue',
                    text=[f'${cost:,.0f}' for cost in server_costs],
                    textposition='outside',
                    textfont=dict(size=9)
                )])
                
                fig.update_layout(
                    title={
                        'text': f'Monthly Cost by Server - Total: ${sum(server_costs):,.0f}',
                        'x': 0.5,
                        'font': {'size': 14, 'color': 'darkblue'}
                    },
                    xaxis_title='Server Name',
                    yaxis_title='Monthly Cost ($)',
                    font=dict(size=10),
                    xaxis={'tickangle': 45, 'tickfont': {'size': 8}},
                    margin=dict(t=50, b=80, l=70, r=50),
                    width=700,
                    height=400
                )
                
                return self.create_plotly_chart_image(fig, width=700, height=400)
                
        except Exception as e:
            print(f"Error creating cost chart: {e}")
            return None
    
    def create_plotly_chart_image(self, fig, width=600, height=400):
        """Convert Plotly figure to image for PDF inclusion with improved quality"""
        try:
            # Convert plotly figure to image bytes with high quality
            img_bytes = fig.to_image(format="png", width=width, height=height, scale=2)
            
            # Create a temporary file to store the image
            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
                tmp_file.write(img_bytes)
                tmp_file.flush()
                
                # Create ReportLab Image object with appropriate sizing
                img = Image(tmp_file.name, width=self.chart_width, height=self.chart_height)
                return img
        except Exception as e:
            print(f"Error creating chart image: {e}")
            return None
    
    def create_improved_technical_table(self, analysis_results, server_specs, env_name):
        """Create an improved technical specifications table"""
        try:
            result = analysis_results.get(env_name, {})
            if 'error' in result:
                return None
            
            # Table data with better formatting
            table_data = [
                ['Component', 'Current (On-Premises)', 'Recommended (AWS)', 'Improvement']
            ]
            
            current_cpu = server_specs.get('cores', 0) if server_specs else 0
            current_ram = server_specs.get('ram', 0) if server_specs else 0
            current_storage = server_specs.get('storage', 0) if server_specs else 0
            
            if 'writer' in result:
                # Aurora/RDS Cluster configuration
                writer = result['writer']
                recommended_cpu = writer.get('actual_vCPUs', 0)
                recommended_ram = writer.get('actual_RAM_GB', 0)
                recommended_storage = result.get('storage_GB', 0)
                
                table_data.extend([
                    ['Writer Instance', 
                     f"{current_cpu} cores", 
                     f"{writer.get('instance_type', 'N/A')}", 
                     f"{(recommended_cpu/max(current_cpu,1)):.1f}x CPU"],
                    ['Writer Memory', 
                     f"{current_ram} GB", 
                     f"{recommended_ram} GB", 
                     f"{(recommended_ram/max(current_ram,1)):.1f}x RAM"],
                    ['Storage Capacity', 
                     f"{current_storage} GB", 
                     f"{recommended_storage} GB", 
                     f"{(recommended_storage/max(current_storage,1)):.1f}x"],
                ])
                
                if result.get('readers'):
                    table_data.append([
                        'Read Replicas', 
                        'None', 
                        f"{len(result['readers'])} instances", 
                        'New capability'
                    ])
            
            else:
                # Standard RDS configuration
                recommended_cpu = result.get('actual_vCPUs', 0)
                recommended_ram = result.get('actual_RAM_GB', 0)
                recommended_storage = result.get('storage_GB', 0)
                
                table_data.extend([
                    ['Instance Type', 
                     'Physical Server', 
                     result.get('instance_type', 'N/A'), 
                     'Cloud Native'],
                    ['CPU Cores', 
                     f"{current_cpu} cores", 
                     f"{recommended_cpu} vCPUs", 
                     f"{(recommended_cpu/max(current_cpu,1)):.1f}x"],
                    ['Memory', 
                     f"{current_ram} GB", 
                     f"{recommended_ram} GB", 
                     f"{(recommended_ram/max(current_ram,1)):.1f}x"],
                    ['Storage', 
                     f"{current_storage} GB", 
                     f"{recommended_storage} GB", 
                     f"{(recommended_storage/max(current_storage,1)):.1f}x"]
                ])
            
            # Create table with improved styling
            col_widths = [1.8*inch, 1.6*inch, 1.8*inch, 1.2*inch]
            tech_table = Table(table_data, colWidths=col_widths, repeatRows=1)
            
            # Enhanced table styling
            tech_table.setStyle(TableStyle([
                # Header styling
                ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('TOPPADDING', (0, 0), (-1, 0), 12),
                
                # Data rows styling
                ('BACKGROUND', (0, 1), (-1, -1), colors.lightblue),
                ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 1), (-1, -1), 9),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.lightblue, colors.white]),
                
                # Left align the first column
                ('ALIGN', (0, 0), (0, -1), 'LEFT'),
                
                # Padding for better readability
                ('LEFTPADDING', (0, 0), (-1, -1), 6),
                ('RIGHTPADDING', (0, 0), (-1, -1), 6),
                ('TOPPADDING', (0, 1), (-1, -1), 8),
                ('BOTTOMPADDING', (0, 1), (-1, -1), 8),
            ]))
            
            return tech_table
            
        except Exception as e:
            print(f"Error creating technical table for {env_name}: {e}")
            return None
    
    def format_ai_insights_text(self, ai_text, max_line_length=80):
        """Format AI insights text for better PDF display"""
        if not ai_text:
            return "No AI insights available."
        
        # Clean up the text
        cleaned_text = ai_text.replace('...', '')
        
        # Split into sentences and paragraphs
        sentences = cleaned_text.split('. ')
        formatted_paragraphs = []
        current_paragraph = ""
        
        for sentence in sentences:
            if len(current_paragraph) + len(sentence) < max_line_length:
                current_paragraph += sentence + ". "
            else:
                if current_paragraph:
                    formatted_paragraphs.append(current_paragraph.strip())
                current_paragraph = sentence + ". "
        
        if current_paragraph:
            formatted_paragraphs.append(current_paragraph.strip())
        
        return formatted_paragraphs[:5]  # Limit to 5 paragraphs for PDF
    
    def generate_improved_pdf_report(self, analysis_results, analysis_mode, server_specs=None, ai_insights=None, transfer_results=None):
        """Generate improved PDF report with better formatting and layout"""
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer, 
            pagesize=A4, 
            rightMargin=0.75*inch, 
            leftMargin=0.75*inch,
            topMargin=1*inch, 
            bottomMargin=1*inch
        )
        
        story = []
        
        try:
            # Title Page
            story.extend(self._create_title_page(analysis_mode, analysis_results))
            story.append(PageBreak())
            
            # Executive Summary with Chart
            story.extend(self._create_executive_summary(analysis_results, analysis_mode, ai_insights))
            story.append(PageBreak())
            
            # Technical Analysis
            story.extend(self._create_technical_analysis(analysis_results, analysis_mode, server_specs))
            story.append(PageBreak())
            
            # Financial Analysis
            story.extend(self._create_financial_analysis(analysis_results, analysis_mode))
            story.append(PageBreak())
            
            # Migration Strategy
            story.extend(self._create_migration_strategy())
            story.append(PageBreak())
            
            # AI Insights (if available)
            if ai_insights:
                story.extend(self._create_ai_insights_section(ai_insights))
                story.append(PageBreak())
            
            # Implementation Roadmap
            story.extend(self._create_implementation_roadmap())
            
            # Build the PDF
            doc.build(story)
            buffer.seek(0)
            return buffer.getvalue()
            
        except Exception as e:
            print(f"Error building improved PDF: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _create_title_page(self, analysis_mode, analysis_results):
        """Create an improved title page"""
        story = []
        
        # Main title
        title_style = self.safe_get_style('ReportTitle', 'Title')
        story.append(Paragraph("AWS RDS Migration & Sizing", title_style))
        story.append(Paragraph("Comprehensive Analysis Report", title_style))
        story.append(Spacer(1, 0.5*inch))
        
        # Analysis type
        analysis_type = "Single Server Analysis" if analysis_mode == 'single' else "Bulk Server Analysis"
        section_style = self.safe_get_style('SectionHeader', 'Heading1')
        story.append(Paragraph(analysis_type, section_style))
        story.append(Spacer(1, 0.3*inch))
        
        # Executive Summary
        if analysis_mode == 'single':
            valid_results = {k: v for k, v in analysis_results.items() if 'error' not in v}
            if valid_results:
                prod_result = valid_results.get('PROD', list(valid_results.values())[0])
                monthly_cost = prod_result.get('total_cost', 0)
                
                summary_items = [
                    f"Monthly Cloud Cost: ${monthly_cost:,.2f}",
                    f"Annual Investment: ${monthly_cost * 12:,.2f}",
                    "Migration Type: Heterogeneous Database Migration",
                    "Estimated Timeline: 12-16 weeks",
                    "Risk Level: Medium (Manageable with proper planning)"
                ]
        else:
            total_servers = len(analysis_results)
            successful_servers = sum(1 for result in analysis_results.values() if 'error' not in result)
            total_monthly_cost = sum(
                result.get('PROD', {}).get('total_cost', 0) 
                for result in analysis_results.values() 
                if 'error' not in result and 'PROD' in result
            )
            
            summary_items = [
                f"Total Servers Analyzed: {total_servers}",
                f"Successful Analyses: {successful_servers}",
                f"Total Monthly Cost: ${total_monthly_cost:,.2f}",
                f"Total Annual Investment: ${total_monthly_cost * 12:,.2f}",
                f"Average Cost per Server: ${total_monthly_cost/max(successful_servers,1):,.2f}/month"
            ]
        
        # Summary box
        summary_style = self.safe_get_style('KeyMetric', 'Normal')
        story.append(Paragraph("<b>Executive Summary</b>", summary_style))
        
        bullet_style = self.safe_get_style('BulletPoint', 'Normal')
        for item in summary_items:
            story.append(Paragraph(f"• {item}", bullet_style))
        
        story.append(Spacer(1, 0.5*inch))
        
        # Report metadata
        body_style = self.safe_get_style('BodyText', 'Normal')
        generation_time = datetime.now().strftime("%B %d, %Y at %H:%M")
        story.append(Paragraph(f"<b>Generated:</b> {generation_time}", body_style))
        story.append(Paragraph("<b>Report Type:</b> Comprehensive Technical & Financial Analysis", body_style))
        story.append(Paragraph("<b>Prepared for:</b> Enterprise Cloud Migration Team", body_style))
        
        return story
    
    def _create_executive_summary(self, analysis_results, analysis_mode, ai_insights):
        """Create improved executive summary with chart"""
        story = []
        
        section_style = self.safe_get_style('SectionHeader', 'Heading1')
        story.append(Paragraph("Executive Summary & Key Findings", section_style))
        story.append(Spacer(1, 12))
        
        # Cost breakdown chart
        cost_chart = self.create_improved_cost_chart(analysis_results, analysis_mode)
        if cost_chart:
            story.append(cost_chart)
            story.append(Spacer(1, 20))
        
        # Key findings
        subsection_style = self.safe_get_style('SubsectionHeader', 'Heading2')
        story.append(Paragraph("Key Findings & Recommendations:", subsection_style))
        
        body_style = self.safe_get_style('BodyText', 'Normal')
        bullet_style = self.safe_get_style('BulletPoint', 'Normal')
        
        if analysis_mode == 'single':
            valid_results = {k: v for k, v in analysis_results.items() if 'error' not in v}
            if valid_results:
                prod_result = valid_results.get('PROD', list(valid_results.values())[0])
                monthly_cost = prod_result.get('total_cost', 0)
                
                findings = [
                    f"<b>Cost Efficiency:</b> Projected monthly operational cost of ${monthly_cost:,.2f} provides significant operational benefits over traditional on-premises infrastructure",
                    "<b>Performance Scaling:</b> AWS RDS configuration will provide improved performance and automatic scaling capabilities",
                    "<b>Risk Mitigation:</b> Multi-AZ deployment ensures 99.95% uptime SLA with automatic failover",
                    "<b>Operational Benefits:</b> Reduced maintenance overhead with managed database services"
                ]
        else:
            successful_servers = sum(1 for result in analysis_results.values() if 'error' not in result)
            total_monthly_cost = sum(
                result.get('PROD', {}).get('total_cost', 0) 
                for result in analysis_results.values() 
                if 'error' not in result and 'PROD' in result
            )
            
            findings = [
                f"<b>Scale Efficiency:</b> {successful_servers} servers successfully analyzed with total monthly cost of ${total_monthly_cost:,.2f}",
                "<b>Migration Approach:</b> Phased migration recommended with 3-5 servers per wave",
                "<b>Cost Optimization:</b> Bulk Reserved Instance purchases can reduce costs by 30-40%",
                "<b>Timeline:</b> Estimated 6-9 months for complete bulk migration with parallel streams"
            ]
        
        for finding in findings:
            story.append(Paragraph(f"• {finding}", bullet_style))
            story.append(Spacer(1, 6))
        
        return story
    
    def _create_technical_analysis(self, analysis_results, analysis_mode, server_specs):
        """Create improved technical analysis section"""
        story = []
        
        section_style = self.safe_get_style('SectionHeader', 'Heading1')
        story.append(Paragraph("Technical Analysis & Performance Assessment", section_style))
        story.append(Spacer(1, 15))
        
        if analysis_mode == 'single':
            valid_results = {k: v for k, v in analysis_results.items() if 'error' not in v}
            
            for env, result in valid_results.items():
                subsection_style = self.safe_get_style('SubsectionHeader', 'Heading2')
                story.append(Paragraph(f"{env} Environment Configuration", subsection_style))
                story.append(Spacer(1, 10))
                
                tech_table = self.create_improved_technical_table(analysis_results, server_specs, env)
                if tech_table:
                    story.append(tech_table)
                    story.append(Spacer(1, 20))
        
        else:
            # Bulk analysis summary table
            story.extend(self._create_bulk_technical_summary(analysis_results, server_specs))
        
        return story
    
    def _create_bulk_technical_summary(self, analysis_results, server_specs):
        """Create bulk technical analysis summary"""
        story = []
        
        # Aggregate statistics
        total_current_cpu = sum(server.get('cpu_cores', 0) for server in server_specs) if server_specs else 0
        total_current_ram = sum(server.get('ram_gb', 0) for server in server_specs) if server_specs else 0
        total_current_storage = sum(server.get('storage_gb', 0) for server in server_specs) if server_specs else 0
        
        total_recommended_cpu = 0
        total_recommended_ram = 0
        total_recommended_storage = 0
        successful_count = 0
        
        for server_results in analysis_results.values():
            if 'error' not in server_results:
                result = server_results.get('PROD', list(server_results.values())[0])
                if 'error' not in result:
                    successful_count += 1
                    if 'writer' in result:
                        writer = result['writer']
                        total_recommended_cpu += writer.get('actual_vCPUs', 0)
                        total_recommended_ram += writer.get('actual_RAM_GB', 0)
                    else:
                        total_recommended_cpu += result.get('actual_vCPUs', 0)
                        total_recommended_ram += result.get('actual_RAM_GB', 0)
                    total_recommended_storage += result.get('storage_GB', 0)
        
        # Create aggregate comparison table
        subsection_style = self.safe_get_style('SubsectionHeader', 'Heading2')
        story.append(Paragraph("Aggregate Resource Comparison", subsection_style))
        story.append(Spacer(1, 10))
        
        aggregate_data = [
            ['Resource Type', 'Current Total', 'Recommended Total', 'Efficiency Gain'],
            ['Total vCPUs', f"{total_current_cpu}", f"{total_recommended_cpu}", f"{((total_recommended_cpu/max(total_current_cpu,1))-1)*100:+.1f}%"],
            ['Total RAM (GB)', f"{total_current_ram:,}", f"{total_recommended_ram:,}", f"{((total_recommended_ram/max(total_current_ram,1))-1)*100:+.1f}%"],
            ['Total Storage (GB)', f"{total_current_storage:,}", f"{total_recommended_storage:,}", f"{((total_recommended_storage/max(total_current_storage,1))-1)*100:+.1f}%"],
            ['Server Count', f"{len(server_specs) if server_specs else 0}", f"{successful_count}", f"{(successful_count/max(len(server_specs) if server_specs else 1,1))*100:.0f}% success"]
        ]
        
        aggregate_table = Table(aggregate_data, colWidths=[2*inch, 1.5*inch, 1.5*inch, 1.5*inch])
        aggregate_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkgreen),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.lightgreen),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ]))
        story.append(aggregate_table)
        
        return story
    
    def _create_financial_analysis(self, analysis_results, analysis_mode):
        """Create improved financial analysis section"""
        story = []
        
        section_style = self.safe_get_style('SectionHeader', 'Heading1')
        story.append(Paragraph("Financial Analysis & ROI Projection", section_style))
        story.append(Spacer(1, 15))
        
        # Calculate costs
        if analysis_mode == 'single':
            valid_results = {k: v for k, v in analysis_results.items() if 'error' not in v}
            if valid_results:
                prod_result = valid_results.get('PROD', list(valid_results.values())[0])
                monthly_cost = prod_result.get('total_cost', 0)
                
                financial_data = [
                    ['Cost Component', 'Monthly', 'Annual', '3-Year Total'],
                    ['AWS Infrastructure', f'${monthly_cost:,.2f}', f'${monthly_cost*12:,.2f}', f'${monthly_cost*36:,.2f}'],
                    ['Migration Costs (One-time)', '-', f'${monthly_cost*2:,.2f}', f'${monthly_cost*2:,.2f}'],
                    ['Training & Support', '-', f'${monthly_cost*0.5:,.2f}', f'${monthly_cost*1.5:,.2f}'],
                    ['Total Investment', f'${monthly_cost:,.2f}', f'${monthly_cost*14.5:,.2f}', f'${monthly_cost*39.5:,.2f}']
                ]
        else:
            total_monthly_cost = sum(
                result.get('PROD', {}).get('total_cost', 0) 
                for result in analysis_results.values() 
                if 'error' not in result and 'PROD' in result
            )
            
            financial_data = [
                ['Cost Component', 'Monthly', 'Annual', '3-Year Total'],
                ['AWS Infrastructure', f'${total_monthly_cost:,.2f}', f'${total_monthly_cost*12:,.2f}', f'${total_monthly_cost*36:,.2f}'],
                ['Migration Costs (One-time)', '-', f'${total_monthly_cost*3:,.2f}', f'${total_monthly_cost*3:,.2f}'],
                ['Training & Support', '-', f'${total_monthly_cost*1:,.2f}', f'${total_monthly_cost*3:,.2f}'],
                ['Total Investment', f'${total_monthly_cost:,.2f}', f'${total_monthly_cost*16:,.2f}', f'${total_monthly_cost*42:,.2f}']
            ]
        
        # Create financial table
        financial_table = Table(financial_data, colWidths=[2.2*inch, 1.4*inch, 1.4*inch, 1.4*inch])
        financial_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkred),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -2), colors.mistyrose),
            ('BACKGROUND', (0, -1), (-1, -1), colors.yellow),
            ('FONTNAME', (0, -1), (-1, -1), 'Helvetica-Bold'),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ]))
        story.append(financial_table)
        
        return story
    
    def _create_migration_strategy(self):
        """Create migration strategy section"""
        story = []
        
        section_style = self.safe_get_style('SectionHeader', 'Heading1')
        story.append(Paragraph("Migration Strategy & Implementation Timeline", section_style))
        story.append(Spacer(1, 15))
        
        subsection_style = self.safe_get_style('SubsectionHeader', 'Heading2')
        story.append(Paragraph("Migration Strategy Overview:", subsection_style))
        
        bullet_style = self.safe_get_style('BulletPoint', 'Normal')
        migration_phases = [
            "<b>Assessment Phase:</b> Complete discovery and compatibility analysis",
            "<b>Planning Phase:</b> Detailed migration planning and resource allocation",
            "<b>Execution Phase:</b> Phased migration with minimal downtime",
            "<b>Validation Phase:</b> Comprehensive testing and performance validation",
            "<b>Optimization Phase:</b> Post-migration tuning and optimization"
        ]
        
        for phase in migration_phases:
            story.append(Paragraph(f"• {phase}", bullet_style))
            story.append(Spacer(1, 6))
        
        return story
    
    def _create_ai_insights_section(self, ai_insights):
        """Create improved AI insights section"""
        story = []
        
        section_style = self.safe_get_style('SectionHeader', 'Heading1')
        story.append(Paragraph("AI-Powered Insights & Recommendations", section_style))
        story.append(Spacer(1, 15))
        
        # AI analysis
        ai_text = ai_insights.get("ai_analysis", "")
        if ai_text:
            formatted_paragraphs = self.format_ai_insights_text(ai_text)
            
            body_style = self.safe_get_style('BodyText', 'Normal')
            for paragraph in formatted_paragraphs:
                story.append(Paragraph(paragraph, body_style))
                story.append(Spacer(1, 10))
        else:
            # Default AI insights
            bullet_style = self.safe_get_style('BulletPoint', 'Normal')
            default_insights = [
                "Migration analysis indicates favorable conditions for AWS RDS migration",
                "Recommended architecture provides optimal balance of performance and cost",
                "Risk assessment suggests manageable migration with proper planning",
                "Cost optimization opportunities identified for long-term efficiency"
            ]
            
            for insight in default_insights:
                story.append(Paragraph(f"• {insight}", bullet_style))
                story.append(Spacer(1, 6))
        
        return story
    
    def _create_implementation_roadmap(self):
        """Create implementation roadmap section"""
        story = []
        
        section_style = self.safe_get_style('SectionHeader', 'Heading1')
        story.append(Paragraph("Implementation Roadmap & Next Steps", section_style))
        story.append(Spacer(1, 15))
        
        subsection_style = self.safe_get_style('SubsectionHeader', 'Heading2')
        story.append(Paragraph("Implementation Roadmap:", subsection_style))
        
        bullet_style = self.safe_get_style('BulletPoint', 'Normal')
        roadmap_phases = [
            "<b>Phase 1 - Planning:</b> Finalize migration strategy and resource allocation",
            "<b>Phase 2 - Preparation:</b> Set up AWS environment and migration tools",
            "<b>Phase 3 - Migration:</b> Execute data and application migration",
            "<b>Phase 4 - Testing:</b> Comprehensive validation and performance testing",
            "<b>Phase 5 - Go-Live:</b> Production cutover and monitoring",
            "<b>Phase 6 - Optimization:</b> Post-migration tuning and optimization"
        ]
        
        for phase in roadmap_phases:
            story.append(Paragraph(f"• {phase}", bullet_style))
            story.append(Spacer(1, 6))
        
        story.append(Spacer(1, 20))
        
        # Success criteria
        story.append(Paragraph("Success Criteria:", subsection_style))
        
        success_criteria = [
            "Zero data loss during migration",
            "Minimal downtime during cutover",
            "Performance meets or exceeds baseline",
            "Cost targets achieved within budget",
            "Team readiness and knowledge transfer complete"
        ]
        
        for criteria in success_criteria:
            story.append(Paragraph(f"• {criteria}", bullet_style))
            story.append(Spacer(1, 6))
        
        return story


# Helper function to use the improved generator
def generate_improved_pdf_report(analysis_results, analysis_mode, server_specs=None, ai_insights=None, transfer_results=None):
    """Helper function with improved formatting and error handling"""
    try:
        print("Creating ImprovedReportGenerator...")
        improved_generator = ImprovedReportGenerator()
        
        print("Validating inputs...")
        if not analysis_results:
            print("Error: No analysis results provided")
            return None
        
        if analysis_mode not in ['single', 'bulk']:
            print(f"Error: Invalid analysis mode '{analysis_mode}'")
            return None
        
        print("Generating improved PDF...")
        pdf_bytes = improved_generator.generate_improved_pdf_report(
            analysis_results=analysis_results,
            analysis_mode=analysis_mode,
            server_specs=server_specs,
            ai_insights=ai_insights,
            transfer_results=transfer_results
        )
        
        if pdf_bytes:
            print(f"Improved PDF generated successfully. Size: {len(pdf_bytes)} bytes")
            return pdf_bytes
        else:
            print("Error: PDF generation returned None")
            return None
        
    except Exception as e:
        print(f"Error in improved PDF generation: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

# ALTERNATIVE: Cached version for better performance
@st.cache_data(ttl=300)  # Cache for 5 minutes
def generate_comprehensive_pdf_report_cached(analysis_results_str, analysis_mode, server_specs_str=None, ai_insights_str=None, transfer_results_str=None):
    """
    Cached version of PDF generator (converts strings back to objects)
    Use this if you're generating the same report multiple times
    """
    import json
    
    try:
        # Convert string parameters back to objects
        analysis_results = json.loads(analysis_results_str) if analysis_results_str else None
        server_specs = json.loads(server_specs_str) if server_specs_str else None
        ai_insights = json.loads(ai_insights_str) if ai_insights_str else None
        transfer_results = json.loads(transfer_results_str) if transfer_results_str else None
        
        # Call the main function
        return generate_improved_pdf_report(
            analysis_results=analysis_results,
            analysis_mode=analysis_mode,
            server_specs=server_specs,
            ai_insights=ai_insights,
            transfer_results=transfer_results
        )
    
    except json.JSONDecodeError as e:
        print(f"Error parsing cached parameters: {e}")
        return None

# USAGE EXAMPLE in your TAB 6:
def example_usage_in_tab6():
    """Example of how to use the improved helper function"""
    
    # Your existing code in TAB 6, replace the PDF generation section with:
    if st.button("📄 Generate Enhanced PDF Report", type="primary", use_container_width=True):
        with st.spinner("Generating Enhanced PDF Report..."):
            try:
                # Prepare inputs with validation
                if not current_analysis_results:
                    st.error("❌ No analysis results available")
                    return
                
                if analysis_mode_for_pdf not in ['single', 'bulk']:
                    st.error(f"❌ Invalid analysis mode: {analysis_mode_for_pdf}")
                    return
                
                # Prepare AI insights (ensure it's not None)
                ai_insights_for_pdf = st.session_state.ai_insights if st.session_state.ai_insights else {
                    "risk_level": "Unknown",
                    "cost_optimization_potential": 0,
                    "ai_analysis": "AI insights were not available during analysis generation."
                }
                
                # Prepare transfer results
                transfer_results_for_pdf = None
                if hasattr(st.session_state, 'transfer_results') and st.session_state.transfer_results:
                    transfer_results_for_pdf = st.session_state.transfer_results
                
                # Generate PDF using the improved helper function
                pdf_bytes = generate_improved_pdf_report(
                    analysis_results=current_analysis_results,
                    analysis_mode=analysis_mode_for_pdf,
                    server_specs=current_server_specs_for_pdf,
                    ai_insights=ai_insights_for_pdf,
                    transfer_results=transfer_results_for_pdf
                )
                
                if pdf_bytes:
                    st.success("✅ Enhanced PDF Report generated successfully!")
                    
                    # Calculate file size
                    file_size_mb = len(pdf_bytes) / (1024 * 1024)
                    st.info(f"📄 PDF Size: {file_size_mb:.2f} MB")
                    
                    # Generate filename with timestamp
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    filename = f"aws_rds_migration_enhanced_{analysis_mode_for_pdf}_{timestamp}.pdf"
                    
                    st.download_button(
                        label="📥 Download Enhanced PDF Report",
                        data=pdf_bytes,
                        file_name=filename,
                        mime="application/pdf",
                        use_container_width=True
                    )
                else:
                    st.error("❌ Failed to generate PDF. Check the error messages above.")
                    
                    # Offer alternative export
                    st.subheader("🔄 Alternative Export Options")
                    if st.button("📊 Export Analysis as JSON (Fallback)", use_container_width=True):
                        export_data = {
                            'analysis_mode': analysis_mode_for_pdf,
                            'analysis_results': current_analysis_results,
                            'server_specs': current_server_specs_for_pdf,
                            'ai_insights': ai_insights_for_pdf,
                            'transfer_results': transfer_results_for_pdf,
                            'generated_at': datetime.now().isoformat()
                        }
                        
                        json_data = json.dumps(export_data, indent=2, default=str)
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                        
                        st.download_button(
                            label="📥 Download JSON Report",
                            data=json_data,
                            file_name=f"analysis_report_{timestamp}.json",
                            mime="application/json",
                            use_container_width=True
                        )
            
            except Exception as e:
                st.error(f"❌ Error in PDF generation process: {str(e)}")
                st.code(traceback.format_exc())

# DEBUG HELPER: Add this temporarily to see what's happening
def debug_pdf_generation(analysis_results, analysis_mode):
    """Debug helper to test PDF generation step by step"""
    print(f"Debug: Analysis mode = {analysis_mode}")
    print(f"Debug: Analysis results type = {type(analysis_results)}")
    print(f"Debug: Analysis results keys = {list(analysis_results.keys()) if isinstance(analysis_results, dict) else 'Not a dict'}")
    
    try:
        generator = ImprovedReportGenerator()
        print("Debug: ComprehensiveReportGenerator created successfully")
        
        # Test if styles are set up correctly
        required_styles = ['ComprehensiveTitle', 'ExecutiveHeader', 'DetailedSectionHeader', 'SubsectionHeader', 'TechnicalSpec']
        missing_styles = []
        for style_name in required_styles:
            try:
                _ = generator.styles[style_name]
            except KeyError:
                missing_styles.append(style_name)
        
        if missing_styles:
            print(f"Debug: Missing styles: {missing_styles}")
        else:
            print("Debug: All required styles are available")
            
        return True
        
    except Exception as e:
        print(f"Debug: Error creating generator: {e}")
        return False

# ================================
# VISUALIZATION FUNCTIONS
# ================================

def create_cost_heatmap(results):
    """Create cost heatmap for environment comparison"""
    if not results:
        return None
    
    valid_results = {k: v for k, v in results.items() if 'error' not in v}
    if not valid_results:
        return None
    
    environments = list(valid_results.keys())
    cost_categories = ['Instance Cost', 'Storage Cost', 'Backup Cost', 'Total Cost']
    
    cost_matrix = []
    for env in environments:
        result = valid_results[env]
        if 'writer' in result and 'readers' in result:
            instance_cost_sum = safe_get(result['cost_breakdown'], 'writer_monthly', 0) + \
                                safe_get(result['cost_breakdown'], 'readers_monthly', 0)
            storage_cost = safe_get(result['cost_breakdown'], 'storage_monthly', 0)
            backup_cost = safe_get(result['cost_breakdown'], 'backup_monthly', 0)
            total_cost = safe_get(result, 'total_cost', 0)
        else:
            cost_breakdown = safe_get(result, 'cost_breakdown', {})
            instance_cost_sum = safe_get(cost_breakdown, 'instance_monthly', safe_get(result, 'instance_cost', 0))
            storage_cost = safe_get(cost_breakdown, 'storage_monthly', safe_get(result, 'storage_cost', 0))
            backup_cost = safe_get(cost_breakdown, 'backup_monthly', safe_get(result, 'storage_cost', 0) * 0.25)
            total_cost = safe_get(result, 'total_cost', 0)

        row = [instance_cost_sum, storage_cost, backup_cost, total_cost]
        cost_matrix.append(row)
    
    cost_matrix = np.array(cost_matrix).T
    
    fig = go.Figure(data=go.Heatmap(
        z=cost_matrix,
        x=environments,
        y=cost_categories,
        colorscale='RdYlBu_r',
        text=[[f'${cost:,.0f}' for cost in row] for row in cost_matrix],
        texttemplate='%{text}',
        textfont={"size": 12},
        colorbar=dict(title="Monthly Cost ($)")
    ))
    
    fig.update_layout(
        title={
            'text': "🔥 Cost Heatmap - All Categories vs Environments",
            'x': 0.5,
            'font': {'size': 16}
        },
        xaxis_title="Environment",
        yaxis_title="Cost Category",
        height=400,
        font=dict(size=12)
    )
    
    return fig

def create_workload_distribution_pie(workload_chars):
    """Create workload characteristics pie chart"""
    if not workload_chars:
        return None
    
    io_mapping = {'read_heavy': 70, 'write_heavy': 30, 'mixed': 50}
    read_pct = io_mapping.get(workload_chars.io_pattern, 50)
    write_pct = 100 - read_pct
    
    fig = go.Figure(data=[go.Pie(
        labels=['Read Operations', 'Write Operations'],
        values=[read_pct, write_pct],
        hole=.3,
        marker_colors=['#36A2EB', '#FF6384']
    )])
    
    fig.update_layout(
        title='📊 Workload I/O Distribution',
        height=350
    )
    
    return fig

def create_bulk_analysis_summary_chart(bulk_results):
    """Create summary chart for bulk analysis results"""
    if not bulk_results:
        return None
    
    server_names = []
    total_costs = []
    instance_types = []
    vcpus = []
    ram_gb = []
    
    for server_name, results in bulk_results.items():
        if 'error' not in results:
            result = results.get('PROD', list(results.values())[0])
            if 'error' not in result:
                server_names.append(server_name)
                total_costs.append(safe_get(result, 'total_cost', 0))
                
                if 'writer' in result:
                    instance_types.append(safe_get_str(result['writer'], 'instance_type', 'Unknown'))
                    vcpus.append(safe_get(result['writer'], 'actual_vCPUs', 0))
                    ram_gb.append(safe_get(result['writer'], 'actual_RAM_GB', 0))
                else:
                    instance_types.append(safe_get_str(result, 'instance_type', 'Unknown'))
                    vcpus.append(safe_get(result, 'actual_vCPUs', 0))
                    ram_gb.append(safe_get(result, 'actual_RAM_GB', 0))
    
    if not server_names:
        return None
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Monthly Cost by Server', 'vCPUs by Server', 'RAM by Server', 'Instance Type Distribution'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"type": "pie"}]]
    )
    
    fig.add_trace(
        go.Bar(x=server_names, y=total_costs, name='Monthly Cost', marker_color='#1f77b4'),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Bar(x=server_names, y=vcpus, name='vCPUs', marker_color='#ff7f0e'),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Bar(x=server_names, y=ram_gb, name='RAM (GB)', marker_color='#2ca02c'),
        row=2, col=1
    )
    
    instance_counts = pd.Series(instance_types).value_counts()
    fig.add_trace(
        go.Pie(labels=instance_counts.index, values=instance_counts.values, name="Instance Types"),
        row=2, col=2
    )
    
    fig.update_layout(
        title_text="📊 Bulk Analysis Summary Dashboard",
        showlegend=False,
        height=600
    )
    
    return fig

def parse_bulk_upload_file(uploaded_file):
    """Parse bulk upload file and extract server specifications"""
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file format. Please upload CSV or Excel files.")
            return None
        
        df.columns = df.columns.str.lower().str.strip()
        
        column_mapping = {
            'server_name': ['server_name', 'servername', 'name', 'hostname', 'server'],
            'cpu_cores': ['cpu_cores', 'cpucores', 'cores', 'cpu', 'processors'],
            'ram_gb': ['ram_gb', 'ramgb', 'ram', 'memory', 'memory_gb'],
            'storage_gb': ['storage_gb', 'storagegb', 'storage', 'disk', 'disk_gb'],
            'peak_cpu_percent': ['peak_cpu_percent', 'peak_cpu', 'cpu_util', 'cpu_utilization', 'max_cpu'],
            'peak_ram_percent': ['peak_ram_percent', 'peak_ram', 'ram_util', 'ram_utilization', 'max_memory'],
            'max_iops': ['max_iops', 'maxiops', 'iops', 'peak_iops'],
            'max_throughput_mbps': ['max_throughput_mbps', 'max_throughput', 'throughput', 'bandwidth'],
            'database_engine': ['database_engine', 'db_engine', 'engine', 'database']
        }
        
        mapped_columns = {}
        for standard_name, possible_names in column_mapping.items():
            for col in df.columns:
                if col in possible_names:
                    mapped_columns[standard_name] = col
                    break
        
        required_columns = ['server_name', 'cpu_cores', 'ram_gb', 'storage_gb']
        missing_columns = [col for col in required_columns if col not in mapped_columns]
        
        if missing_columns:
            st.error(f"Missing required columns: {missing_columns}")
            st.info("Required columns: Server Name, CPU Cores, RAM (GB), Storage (GB)")
            return None
        
        servers = []
        for idx, row in df.iterrows():
            try:
                server = {
                    'server_name': str(row[mapped_columns['server_name']]).strip(),
                    'cpu_cores': int(float(row[mapped_columns['cpu_cores']])),
                    'ram_gb': int(float(row[mapped_columns['ram_gb']])),
                    'storage_gb': int(float(row[mapped_columns['storage_gb']])),
                    'peak_cpu_percent': int(float(row.get(mapped_columns.get('peak_cpu_percent', ''), 75))),
                    'peak_ram_percent': int(float(row.get(mapped_columns.get('peak_ram_percent', ''), 80))),
                    'max_iops': int(float(row.get(mapped_columns.get('max_iops', ''), 1000))),
                    'max_throughput_mbps': int(float(row.get(mapped_columns.get('max_throughput_mbps', ''), 125))),
                    'database_engine': str(row.get(mapped_columns.get('database_engine', ''), 'oracle-ee')).strip().lower()
                }
                
                if server['cpu_cores'] <= 0 or server['ram_gb'] <= 0 or server['storage_gb'] <= 0:
                    st.warning(f"Invalid data for server {server['server_name']} at row {idx + 1}. Skipping.")
                    continue
                
                servers.append(server)
                
            except (ValueError, TypeError) as e:
                st.warning(f"Error parsing row {idx + 1}: {e}. Skipping.")
                continue
        
        if not servers:
            st.error("No valid server data found in the uploaded file.")
            return None
        
        st.success(f"Successfully parsed {len(servers)} servers from the uploaded file.")
        return servers
        
    except Exception as e:
        st.error(f"Error parsing file: {e}")
        return None

# ================================
# INITIALIZATION
# ================================

# Check authentication before showing the app
if not check_authentication():
    show_login_form()
    st.stop()

# Enhanced Custom CSS
st.markdown("""
<style>
    .main > div {
        padding-top: 1rem;
    }
    
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        color: white;
        margin-bottom: 0.5rem;
    }
    
    .metric-label {
        font-size: 1rem;
        color: rgba(255, 255, 255, 0.9);
        font-weight: 500;
    }
    
    .status-card {
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid;
    }
    
    .status-success {
        background: #d4edda;
        border-left-color: #28a745;
        color: #155724;
    }
    
    .status-warning {
        background: #fff3cd;
        border-left-color: #ffc107;
        color: #856404;
    }
    
    .status-error {
        background: #f8d7da;
        border-left-color: #dc3545;
        color: #721c24;
    }
    
    .status-info {
        background: #d1ecf1;
        border-left-color: #17a2b8;
        color: #0c5460;
    }
    
    .migration-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
    }
    
    .ai-insight-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
    }    
    .advisory-box {
        background: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #007bff;
    }    
    .phase-timeline {
        background: linear-gradient(90deg, #a8edea 0%, #fed6e3 100%);
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }    
    .chart-container {
        background: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }    
    .writer-box {
        background: #e3f2fd;
        border: 2px solid #2196f3;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }    
    .reader-box {
        background: #f3e5f5;
        border: 2px solid #9c27b0;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }    
    .spec-section {
        background: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
    }    
    .bulk-upload-zone {
        border: 2px dashed #007bff;
        border-radius: 8px;
        padding: 2rem;
        text-align: center;
        background: #f8f9ff;
        margin: 1rem 0;
    }    
    .server-summary-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'calculator' not in st.session_state:
    st.session_state.calculator = None
if 'results' not in st.session_state:
    st.session_state.results = None
if 'ai_insights' not in st.session_state:
    st.session_state.ai_insights = None
if 'migration_configured' not in st.session_state:
    st.session_state.migration_configured = False
if 'user_claude_api_key_input' not in st.session_state:
    st.session_state.user_claude_api_key_input = ""
if 'source_engine' not in st.session_state:
    st.session_state.source_engine = None
if 'target_engine' not in st.session_state:
    st.session_state.target_engine = None
if 'deployment_option' not in st.session_state:
    st.session_state.deployment_option = "Multi-AZ"
if 'bulk_results' not in st.session_state:
    st.session_state.bulk_results = {}
if 'on_prem_servers' not in st.session_state:
    st.session_state.on_prem_servers = []
if 'bulk_upload_data' not in st.session_state:
    st.session_state.bulk_upload_data = None
if 'current_analysis_mode' not in st.session_state:
    st.session_state.current_analysis_mode = 'single'
if 'firebase_app' not in st.session_state:
    st.session_state.firebase_app = None
if 'firebase_auth' not in st.session_state:
    st.session_state.firebase_auth = None
if 'firebase_db' not in st.session_state:
    st.session_state.firebase_db = None
if 'transfer_results' not in st.session_state:
    st.session_state.transfer_results = None
if 'transfer_data_size' not in st.session_state:
    st.session_state.transfer_data_size = 0
if 'user_email' not in st.session_state:
    st.session_state.user_email = ""

# Initialize Firebase
if st.session_state.firebase_app is None:
    st.session_state.firebase_app, st.session_state.firebase_auth, st.session_state.firebase_db = initialize_firebase()

# Initialize components
@st.cache_resource
def initialize_components():
    """Initialize the pricing API"""
    try:
        pricing_api = EnhancedAWSPricingAPI()
        return pricing_api
    except Exception as e:
        st.error(f"Error initializing static components: {e}")
        return None

pricing_api = initialize_components()
if not pricing_api:
    st.error("Failed to initialize required components")
    st.stop()
    
@st.cache_resource
def initialize_transfer_calculator():
    """Initialize the data transfer calculator"""
    try:
        return DataTransferCalculator()
    except Exception as e:
        st.error(f"Error initializing transfer calculator: {e}")
        return None

transfer_calculator = initialize_transfer_calculator()

# Initialize calculator
if st.session_state.calculator is None:
    claude_api_key = None
    if st.session_state.user_claude_api_key_input:
        claude_api_key = st.session_state.user_claude_api_key_input
    elif "anthropic" in st.secrets and "ANTHROPIC_API_KEY" in st.secrets["anthropic"]:
        claude_api_key = st.secrets["anthropic"]["ANTHROPIC_API_KEY"]
    else:
        claude_api_key = os.environ.get('ANTHROPIC_API_KEY')
    
    st.session_state.calculator = EnhancedRDSSizingCalculator(
        anthropic_api_key=claude_api_key,
        use_real_time_pricing=True
    )

# ================================
# MAIN APPLICATION
# ================================

# Header
st.title("🚀 Enterprise AWS RDS Migration & Sizing Tool")
st.markdown("**AI-Powered Analysis • Homogeneous & Heterogeneous Migrations • Real-time AWS Pricing • Advanced Analytics**")

# API Key input
st.subheader("🤖 AI Integration (Anthropic Claude API Key)")
st.session_state.user_claude_api_key_input = st.text_input(
    "Enter your Anthropic API Key (optional)",
    type="password",
    value=st.session_state.user_claude_api_key_input,
    help="Provide your Anthropic API key here to enable AI-powered insights."
)
st.markdown("---")

# Show user info in sidebar
show_user_info()

# Display System Status
st.sidebar.subheader("System Status")
if st.session_state.firebase_app:
    st.sidebar.success("🔥 Firebase Connected")
    st.sidebar.write(f"**Project:** {st.session_state.firebase_app.project_id}")
else:
    st.sidebar.warning("Firebase not connected")

st.markdown("---")

# Inject mock data for testing PDF generation (remove in production)
# This mock data is ONLY injected if bulk_results is empty.
if 'bulk_results' not in st.session_state or not st.session_state.bulk_results:
    if 'server1' not in st.session_state.bulk_results:
        st.session_state.bulk_results = {
            'server1': {
                'PROD': {
                    'total_cost': 1500,
                    'instance_type': 'db.m5.large',
                    'actual_vCPUs': 2,
                    'actual_RAM_GB': 8,
                    'storage_GB': 100,
                    'cost_breakdown': {
                        'instance_monthly': 1200,
                        'storage_monthly': 200,
                        'backup_monthly': 100
                    },
                    'writer': {
                        'instance_type': 'db.m5.large',
                        'actual_vCPUs': 2,
                        'actual_RAM_GB': 8
                    }
                }
            }
        }
        st.session_state.on_prem_servers = [{
            'server_name': 'server1',
            'cpu_cores': 2,
            'ram_gb': 8,
            'storage_gb': 100,
            'peak_cpu_percent': 75,
            'peak_ram_percent': 80,
            'max_iops': 1000,
            'max_throughput_mbps': 125,
            'database_engine': 'oracle-ee'
        }]
    
    # Also inject a dummy AI insight for mock data if none exists
    if not st.session_state.ai_insights:
        st.session_state.ai_insights = {
            "risk_level": "Medium",
            "cost_optimization_potential": 0.15,
            "recommended_writers": 1,
            "recommended_readers": 1,
            "ai_analysis": "This is a mock AI analysis for the single server. It suggests optimizing storage and considering a multi-AZ deployment for high availability. The database workload seems balanced with a slight read bias. Further analysis with historical performance data would refine recommendations."
        }
    
    # Also inject a dummy transfer result for mock data if none exists
    if not st.session_state.transfer_results:
        st.session_state.transfer_results = {
            'datasync_dx': TransferMethodResult(
                recommended_method='AWS DataSync (Direct Connect)',
                transfer_time_hours=10.5,
                transfer_time_days=0.4,
                total_cost=50.0,
                bandwidth_utilization=90.0,
                estimated_downtime_hours=0.1,
                cost_breakdown={'data_transfer': 25.0, 'datasync_task': 25.0}
            ),
            'datasync_internet': TransferMethodResult(
                recommended_method='AWS DataSync (Internet)',
                transfer_time_hours=24.0,
                transfer_time_days=1.0,
                total_cost=30.0,
                bandwidth_utilization=70.0,
                estimated_downtime_hours=0.5,
                cost_breakdown={'data_transfer': 10.0, 'datasync_task': 20.0}
            )
        }
        st.session_state.transfer_data_size = 500

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["🎯 Migration Planning", "🖥️ Server Specifications", "📊 Sizing Analysis", "💰 Financial Analysis", "🤖 AI Insights", "📋 Reports"])

# ================================
# TAB 1: MIGRATION PLANNING
# ================================

with tab1:
    st.header("Migration Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔄 Migration Type")
        
        source_engine_selection = st.selectbox(
            "Source Database Engine",
            ["oracle-ee", "oracle-se", "oracle-se1", "oracle-se2", "sqlserver-ee", "sqlserver-se", 
             "mysql", "postgres", "mariadb"],
            index=0 if st.session_state.source_engine is None else ["oracle-ee", "oracle-se", "oracle-se1", "oracle-se2", "sqlserver-ee", "sqlserver-se", 
             "mysql", "postgres", "mariadb"].index(st.session_state.source_engine),
            key="source_engine_select"
        )
        
        target_engine_selection = st.selectbox(
            "Target AWS Database Engine",
            ["postgres", "aurora-postgresql", "aurora-mysql", "mysql", "oracle-ee", "oracle-se2", 
             "sqlserver-ee", "sqlserver-se"],
            index=0 if st.session_state.target_engine is None else ["postgres", "aurora-postgresql", "aurora-mysql", "mysql", "oracle-ee", "oracle-se2", 
             "sqlserver-ee", "sqlserver-se"].index(st.session_state.target_engine),
            key="target_engine_select"
        )
        
        if source_engine_selection.split('-')[0] == target_engine_selection.split('-')[0]:
            migration_type_display = "Homogeneous"
            migration_color = "success"
        else:
            migration_type_display = "Heterogeneous"
            migration_color = "warning"
        
        st.markdown(f"""
        <div class="status-card status-{migration_color}">
            <strong>Migration Type: {migration_type_display}</strong><br>
            {source_engine_selection} → {target_engine_selection}
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.subheader("⚙️ Workload Characteristics")
        
        cpu_pattern = st.selectbox("CPU Utilization Pattern", ["steady", "bursty", "peak_hours"], key="cpu_pattern_select")
        memory_pattern = st.selectbox("Memory Usage Pattern", ["steady", "high_variance", "growing"], key="memory_pattern_select")
        io_pattern = st.selectbox("I/O Pattern", ["read_heavy", "write_heavy", "mixed"], key="io_pattern_select")
        connection_count = st.number_input("Typical Connection Count", min_value=10, max_value=10000, value=100, step=10, key="connection_count_input")
        transaction_volume = st.selectbox("Transaction Volume", ["low", "medium", "high", "very_high"], index=1, key="transaction_volume_select")
        analytical_workload = st.checkbox("Analytical/Reporting Workload", key="analytical_workload_checkbox")
    
    st.subheader("☁️ AWS Configuration")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        region = st.selectbox("AWS Region", ["us-east-1", "us-west-1", "us-west-2", "eu-west-1", "eu-central-1", "ap-southeast-1"], key="region_select")
    
    with col2:
        st.session_state.deployment_option = st.selectbox(
            "Deployment Option", 
            ["Single-AZ", "Multi-AZ", "Multi-AZ Cluster", "Aurora Global", "Serverless"], 
            index=["Single-AZ", "Multi-AZ", "Multi-AZ Cluster", "Aurora Global", "Serverless"].index(st.session_state.deployment_option),
            key="deployment_option_select"
        )
    
    with col3:
        storage_type = st.selectbox("Storage Type", ["gp3", "gp2", "io1", "io2", "aurora"], key="storage_type_select")
    
    st.session_state.source_engine = source_engine_selection
    st.session_state.target_engine = target_engine_selection
    st.session_state.region = region
    st.session_state.storage_type = storage_type

    if st.button("🎯 Configure Migration", type="primary", use_container_width=True):
        with st.spinner("Configuring migration parameters..."):
            try:
                # Re-initialize calculator with updated API key if provided
                claude_api_key_current = None
                if st.session_state.user_claude_api_key_input:
                    claude_api_key_current = st.session_state.user_claude_api_key_input
                elif "anthropic" in st.secrets and "ANTHROPIC_API_KEY" in st.secrets["anthropic"]:
                    claude_api_key_current = st.secrets["anthropic"]["ANTHROPIC_API_KEY"]
                else:
                    claude_api_key_current = os.environ.get('ANTHROPIC_API_KEY')

                # Only re-initialize if the key has changed or calculator is None
                if st.session_state.calculator is None or \
                   (hasattr(st.session_state.calculator, 'anthropic_api_key') and \
                    st.session_state.calculator.anthropic_api_key != claude_api_key_current) or \
                   (not hasattr(st.session_state.calculator, 'anthropic_api_key') and claude_api_key_current):
                    st.session_state.calculator = EnhancedRDSSizingCalculator(
                        anthropic_api_key=claude_api_key_current,
                        use_real_time_pricing=True
                    )

                workload_chars = WorkloadCharacteristics(
                    cpu_utilization_pattern=cpu_pattern,
                    memory_usage_pattern=memory_pattern,
                    io_pattern=io_pattern,
                    connection_count=connection_count,
                    transaction_volume=transaction_volume,
                    analytical_workload=analytical_workload
                )
                
                st.session_state.calculator.set_migration_parameters(
                    source_engine_selection, target_engine_selection, workload_chars
                )
                
                st.session_state.migration_configured = True
                
                migration_profile = st.session_state.calculator.migration_profile
                
                st.markdown(f"""
                <div class="migration-card">
                    <h3>🎯 Migration Configuration Complete</h3>
                    <strong>Migration Type:</strong> {migration_profile.migration_type.value.title()}<br>
                    <strong>Complexity Factor:</strong> {migration_profile.complexity_factor:.1f}x<br>
                    <strong>Feature Compatibility:</strong> {migration_profile.feature_compatibility*100:.1f}%<br>
                    <strong>Recommended Sizing Buffer:</strong> {migration_profile.recommended_sizing_buffer*100:.1f}%
                </div>
                """, unsafe_allow_html=True)
                
                st.success("✅ Migration configured successfully! Proceed to Server Specifications.")
                
            except Exception as e:
                st.error(f"❌ Error configuring migration: {str(e)}")


# ================================
# TAB 2: SERVER SPECIFICATIONS (CORRECTED)
# ================================

with tab2:
    st.header("🖥️ On-Premises Server Specifications")
    
    if not st.session_state.migration_configured:
        st.warning("⚠️ Please configure migration settings in the Migration Planning tab first.")
    else:
        st.subheader("📋 Analysis Mode")
        analysis_mode = st.radio(
            "Choose Analysis Mode",
            ["Single Server Analysis", "Bulk Server Analysis"],
            horizontal=True,
            key="analysis_mode_radio"
        )
        
        st.session_state.current_analysis_mode = 'single' if analysis_mode == "Single Server Analysis" else 'bulk'
        
        if st.session_state.current_analysis_mode == 'single':
            # Single Server Specifications
            st.subheader("🖥️ Single Server Configuration")

            with st.expander("📋 Server Information", expanded=True):
                col1, col2, col3 = st.columns(3)
            
                with col1:
                    manual_server_name = st.text_input("Server Name/Hostname", value="PROD-DB-01", key="manual_server_name_input")
                    manual_cores = st.number_input("CPU Cores", min_value=1, max_value=128, value=8, key="manual_cores_input")
                    manual_ram = st.number_input("RAM (GB)", min_value=1, max_value=1024, value=32, key="manual_ram_input")
                
                with col2:
                    manual_storage = st.number_input("Storage (GB)", min_value=10, value=250, key="manual_storage_input")
                    manual_cpu_util = st.number_input("Peak CPU (%)", min_value=1, max_value=100, value=70, key="manual_cpu_util_input")
                    manual_ram_util = st.number_input("Peak RAM (%)", min_value=1, max_value=100, value=75, key="manual_ram_util_input")
                
                with col3:
                    manual_iops = st.number_input("Max IOPS", min_value=100, value=2500, key="manual_iops_input")
                    manual_throughput = st.number_input("Max Throughput (MB/s)", min_value=10, value=125, key="manual_throughput_input")
                    manual_engine = st.selectbox("Database Engine", ["oracle-ee", "oracle-se", "mysql", "postgres"], key="manual_engine_select")
            
            # Add to bulk list button
            if st.button("➕ Add Server to Bulk List", use_container_width=True):
                if manual_server_name:
                    new_server = {
                        'server_name': manual_server_name,
                        'cpu_cores': manual_cores,
                        'ram_gb': manual_ram,
                        'storage_gb': manual_storage,
                        'peak_cpu_percent': manual_cpu_util,
                        'peak_ram_percent': manual_ram_util,
                        'max_iops': manual_iops,
                        'max_throughput_mbps': manual_throughput,
                        'database_engine': manual_engine
                    }
                    
                    if 'on_prem_servers' not in st.session_state:
                        st.session_state.on_prem_servers = []
                    
                    st.session_state.on_prem_servers.append(new_server)
                    st.success(f"✅ Added {manual_server_name} to bulk analysis list")
                    st.rerun()
                else:
                    st.error("Please provide a server name")
            
            # Single server analysis button
            if st.button("🔍 Analyze This Server (Single Mode)", type="primary", use_container_width=True):
                if manual_server_name and manual_cores and manual_ram:
                    # Store current server spec for single analysis
                    st.session_state.current_server_spec = {
                        'server_name': manual_server_name,
                        'cores': manual_cores,
                        'cpu_ghz': 2.4,  # Default value
                        'ram': manual_ram,
                        'ram_type': 'DDR4',  # Default value
                        'storage': manual_storage,
                        'storage_type': 'SSD',  # Default value
                        'cpu_util': manual_cpu_util,
                        'ram_util': manual_ram_util,
                        'max_iops': manual_iops,
                        'max_throughput_mbps': manual_throughput,
                        'max_connections': 500,  # Default value
                        'growth_rate': 20,  # Default 20% growth
                        'years': 3,  # Default 3 years
                        'enable_encryption': True,
                        'enable_perf_insights': True,
                        'enable_enhanced_monitoring': False,
                        'monthly_transfer_gb': 100  # Default value
                    }
                    st.success(f"✅ Server {manual_server_name} configured for single analysis")
                    st.info("👉 Go to the 'Sizing Analysis' tab to generate recommendations")
                else:
                    st.error("Please provide server name, CPU cores, and RAM")
        
        else:
            # BULK SERVER ANALYSIS MODE
            # Replace the bulk analysis section in TAB 3 with this corrected version
# This should go in the "else" block for bulk analysis mode

# Bulk Server Analysis
            # BULK SERVER ANALYSIS MODE
            st.subheader("📊 Bulk Server Analysis")
            
            # Bulk upload options
            st.markdown("### 📁 Upload Server Specifications")
            
            upload_method = st.radio(
                "Choose Upload Method",
                ["📄 Upload CSV/Excel File", "✍️ Manual Entry"],
                horizontal=True,
                key="bulk_upload_method"
            )
            
            if upload_method == "📄 Upload CSV/Excel File":
                # File Upload Section
                st.markdown("""
                <div class="bulk-upload-zone">
                    <h4>📂 Upload Server Specifications File</h4>
                    <p>Upload a CSV or Excel file containing your server specifications</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Show required columns
                with st.expander("📋 Required File Format", expanded=False):
                    st.markdown("""
                    **Required Columns:**
                    - `server_name` - Server hostname or identifier
                    - `cpu_cores` - Number of CPU cores
                    - `ram_gb` - RAM in GB
                    - `storage_gb` - Storage in GB
                    
                    **Optional Columns:**
                    - `peak_cpu_percent` - Peak CPU utilization (default: 75%)
                    - `peak_ram_percent` - Peak RAM utilization (default: 80%)
                    - `max_iops` - Maximum IOPS (default: 1000)
                    - `max_throughput_mbps` - Maximum throughput in MB/s (default: 125)
                    - `database_engine` - Database engine (default: oracle-ee)
                    """)
                    
                    # Sample data download
                    sample_data = pd.DataFrame([
                        {
                            'server_name': 'PROD-DB-01',
                            'cpu_cores': 8,
                            'ram_gb': 32,
                            'storage_gb': 500,
                            'peak_cpu_percent': 75,
                            'peak_ram_percent': 80,
                            'max_iops': 2500,
                            'max_throughput_mbps': 125,
                            'database_engine': 'oracle-ee'
                        },
                        {
                            'server_name': 'TEST-DB-01',
                            'cpu_cores': 4,
                            'ram_gb': 16,
                            'storage_gb': 250,
                            'peak_cpu_percent': 60,
                            'peak_ram_percent': 70,
                            'max_iops': 1500,
                            'max_throughput_mbps': 100,
                            'database_engine': 'mysql'
                        },
                        {
                            'server_name': 'DEV-DB-01',
                            'cpu_cores': 2,
                            'ram_gb': 8,
                            'storage_gb': 100,
                            'peak_cpu_percent': 50,
                            'peak_ram_percent': 60,
                            'max_iops': 1000,
                            'max_throughput_mbps': 75,
                            'database_engine': 'postgres'
                        }
                    ])
                    
                    sample_csv = sample_data.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Sample CSV Template",
                        data=sample_csv,
                        file_name="server_specs_template.csv",
                        mime="text/csv",
                        help="Download this template and fill in your server data"
                    )
                
                # File uploader
                uploaded_file = st.file_uploader(
                    "Choose CSV or Excel file",
                    type=['csv', 'xlsx', 'xls'],
                    help="Upload a file containing your server specifications",
                    key="bulk_upload_file"
                )
                
                if uploaded_file is not None:
                    with st.spinner("📖 Parsing uploaded file..."):
                        try:
                            parsed_servers = parse_bulk_upload_file(uploaded_file)
                            
                            if parsed_servers:
                                st.session_state.bulk_upload_data = parsed_servers
                                st.session_state.on_prem_servers = parsed_servers
                                
                                st.success(f"✅ Successfully parsed {len(parsed_servers)} servers from {uploaded_file.name}")
                                
                                # Show preview of uploaded data
                                st.subheader("📊 Uploaded Server Data Preview")
                                preview_df = pd.DataFrame(parsed_servers)
                                st.dataframe(preview_df, use_container_width=True)
                                
                                # Data validation summary
                                col1, col2, col3, col4 = st.columns(4)
                                
                                with col1:
                                    total_cores = sum([server['cpu_cores'] for server in parsed_servers])
                                    st.metric("Total CPU Cores", total_cores)
                                
                                with col2:
                                    total_ram = sum([server['ram_gb'] for server in parsed_servers])
                                    st.metric("Total RAM (GB)", total_ram)
                                
                                with col3:
                                    total_storage = sum([server['storage_gb'] for server in parsed_servers])
                                    st.metric("Total Storage (GB)", f"{total_storage:,}")
                                
                                with col4:
                                    unique_engines = len(set([server['database_engine'] for server in parsed_servers]))
                                    st.metric("DB Engine Types", unique_engines)
                                
                            else:
                                st.error("❌ Failed to parse the uploaded file. Please check the format and try again.")
                                
                        except Exception as e:
                            st.error(f"❌ Error processing file: {str(e)}")
                            st.info("💡 Please ensure your file matches the required format.")
            
            else:
                # Manual Entry for Bulk
                st.markdown("### ✍️ Manual Server Entry")
                
                with st.form("manual_bulk_entry"):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        bulk_server_name = st.text_input("Server Name", placeholder="e.g., PROD-DB-02")
                        bulk_cores = st.number_input("CPU Cores", min_value=1, max_value=128, value=4)
                        bulk_ram = st.number_input("RAM (GB)", min_value=1, max_value=1024, value=16)
                    
                    with col2:
                        bulk_storage = st.number_input("Storage (GB)", min_value=10, value=250)
                        bulk_cpu_util = st.number_input("Peak CPU (%)", min_value=1, max_value=100, value=70)
                        bulk_ram_util = st.number_input("Peak RAM (%)", min_value=1, max_value=100, value=75)
                    
                    with col3:
                        bulk_iops = st.number_input("Max IOPS", min_value=100, value=1500)
                        bulk_throughput = st.number_input("Max Throughput (MB/s)", min_value=10, value=100)
                        bulk_engine = st.selectbox("Database Engine", 
                                                   ["oracle-ee", "oracle-se", "mysql", "postgres", "sqlserver-ee"], 
                                                   index=0)
                    
                    submitted = st.form_submit_button("➕ Add Server to Bulk List", use_container_width=True)
                    
                    if submitted:
                        if bulk_server_name:
                            new_server = {
                                'server_name': bulk_server_name,
                                'cpu_cores': bulk_cores,
                                'ram_gb': bulk_ram,
                                'storage_gb': bulk_storage,
                                'peak_cpu_percent': bulk_cpu_util,
                                'peak_ram_percent': bulk_ram_util,
                                'max_iops': bulk_iops,
                                'max_throughput_mbps': bulk_throughput,
                                'database_engine': bulk_engine
                            }
                            
                            if 'on_prem_servers' not in st.session_state:
                                st.session_state.on_prem_servers = []
                            
                            # Check for duplicate server names
                            existing_names = [s['server_name'] for s in st.session_state.on_prem_servers]
                            if bulk_server_name in existing_names:
                                st.error(f"❌ Server name '{bulk_server_name}' already exists. Please use a unique name.")
                            else:
                                st.session_state.on_prem_servers.append(new_server)
                                st.success(f"✅ Added {bulk_server_name} to bulk analysis list")
                                st.rerun()
                        else:
                            st.error("❌ Please provide a server name")

            # Display current server list
            if st.session_state.on_prem_servers:
                st.subheader(f"📊 Current Server List ({len(st.session_state.on_prem_servers)} servers)")
                
                # Create DataFrame for display
                display_df = pd.DataFrame(st.session_state.on_prem_servers)
                
                # Add edit capabilities
                edited_df = st.data_editor(
                    display_df,
                    use_container_width=True,
                    num_rows="dynamic",
                    key="server_list_editor",
                    column_config={
                        "server_name": st.column_config.TextColumn("Server Name", required=True),
                        "cpu_cores": st.column_config.NumberColumn("CPU Cores", min_value=1, max_value=128),
                        "ram_gb": st.column_config.NumberColumn("RAM (GB)", min_value=1, max_value=1024),
                        "storage_gb": st.column_config.NumberColumn("Storage (GB)", min_value=10),
                        "peak_cpu_percent": st.column_config.NumberColumn("Peak CPU %", min_value=1, max_value=100),
                        "peak_ram_percent": st.column_config.NumberColumn("Peak RAM %", min_value=1, max_value=100),
                        "max_iops": st.column_config.NumberColumn("Max IOPS", min_value=100),
                        "max_throughput_mbps": st.column_config.NumberColumn("Throughput MB/s", min_value=10),
                        "database_engine": st.column_config.SelectboxColumn("DB Engine", 
                                                                            options=["oracle-ee", "oracle-se", "mysql", "postgres", "sqlserver-ee"])
                    }
                )
                
                # Update session state with edited data
                if not edited_df.equals(display_df):
                    st.session_state.on_prem_servers = edited_df.to_dict('records')
                    st.success("✅ Server list updated!")
                
                # Server list management
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    if st.button("🗑️ Clear All Servers", use_container_width=True):
                        st.session_state.on_prem_servers = []
                        st.session_state.bulk_upload_data = None
                        st.session_state.bulk_results = {}
                        st.success("✅ All servers cleared")
                        st.rerun()
                
                with col2:
                    # Export current list
                    current_csv = pd.DataFrame(st.session_state.on_prem_servers).to_csv(index=False)
                    st.download_button(
                        label="📥 Export Server List",
                        data=current_csv,
                        file_name=f"server_list_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                
                with col3:
                    if st.button("🔄 Validate All Servers", use_container_width=True):
                        validation_errors = []
                        for i, server in enumerate(st.session_state.on_prem_servers):
                            if not server.get('server_name'):
                                validation_errors.append(f"Row {i+1}: Missing server name")
                            if server.get('cpu_cores', 0) <= 0:
                                validation_errors.append(f"Row {i+1}: Invalid CPU cores")
                            if server.get('ram_gb', 0) <= 0:
                                validation_errors.append(f"Row {i+1}: Invalid RAM")
                            if server.get('storage_gb', 0) <= 0:
                                validation_errors.append(f"Row {i+1}: Invalid storage")
                        
                        if validation_errors:
                            st.error(f"❌ Validation errors found:")
                            for error in validation_errors:
                                st.write(f"• {error}")
                        else:
                            st.success("✅ All servers validated successfully!")
                
                with col4:
                    # Show summary stats
                    if st.button("📊 Show Summary", use_container_width=True):
                        st.session_state.show_summary = not st.session_state.get('show_summary', False)
                
                # Summary statistics
                if st.session_state.get('show_summary', False):
                    st.subheader("📈 Server List Summary")
                    
                    total_cores = sum([server['cpu_cores'] for server in st.session_state.on_prem_servers])
                    total_ram = sum([server['ram_gb'] for server in st.session_state.on_prem_servers])
                    total_storage = sum([server['storage_gb'] for server in st.session_state.on_prem_servers])
                    avg_cpu_util = sum([server['peak_cpu_percent'] for server in st.session_state.on_prem_servers]) / len(st.session_state.on_prem_servers)
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Total CPU Cores", total_cores)
                    
                    with col2:
                        st.metric("Total RAM (GB)", f"{total_ram:,}")
                    
                    with col3:
                        st.metric("Total Storage (GB)", f"{total_storage:,}")
                    
                    with col4:
                        st.metric("Avg CPU Utilization", f"{avg_cpu_util:.1f}%")
                    
                    # Engine distribution
                    engine_counts = {}
                    for server in st.session_state.on_prem_servers:
                        engine = server['database_engine']
                        engine_counts[engine] = engine_counts.get(engine, 0) + 1
                    
                    st.markdown("**Database Engine Distribution:**")
                    for engine, count in engine_counts.items():
                        st.write(f"• {engine}: {count} servers")

                # Replace the problematic bulk analysis section in TAB 3 (around lines 3050-3150) with this corrected version:

## ================================
# TAB 3: SIZING ANALYSIS (CORRECTED VERSION)
# ================================

with tab3:
    st.header("📊 Sizing Analysis & Recommendations")
    
    if not st.session_state.migration_configured:
        st.warning("⚠️ Please configure migration settings in the Migration Planning tab first.")
    elif st.session_state.current_analysis_mode == 'single' and 'current_server_spec' not in st.session_state:
        st.warning("⚠️ Please configure server specifications in the Server Specifications tab first.")
    elif st.session_state.current_analysis_mode == 'bulk' and not st.session_state.on_prem_servers:
        st.warning("⚠️ Please upload or add server specifications for bulk analysis.")
    else:
        calculator = st.session_state.calculator
        
        if st.session_state.current_analysis_mode == 'single':
            # ================================
            # SINGLE SERVER ANALYSIS
            # ================================
            st.subheader("🖥️ Single Server Analysis")
            
            server_spec = st.session_state.current_server_spec
            
            st.markdown(f"""
            <div class="server-summary-card">
                <h4>🔍 Analyzing: {server_spec['server_name']}</h4>
                <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin-top: 1rem;">
                    <div><strong>CPU:</strong> {server_spec['cores']} cores @ {server_spec.get('cpu_ghz', 2.4)}GHz</div>
                    <div><strong>RAM:</strong> {server_spec['ram']}GB {server_spec.get('ram_type', 'DDR4')}</div>
                    <div><strong>Storage:</strong> {server_spec['storage']}GB {server_spec.get('storage_type', 'SSD')}</div>
                    <div><strong>IOPS:</strong> {server_spec.get('max_iops', 1000):,} peak</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                if st.button("🚀 Generate Sizing Recommendations", type="primary", use_container_width=True):
                    with st.spinner("🔄 Analyzing workload and generating recommendations..."):
                        start_time = time.time()
                        
                        try:
                            inputs = {
                                "region": st.session_state.region,
                                "target_engine": st.session_state.target_engine,
                                "source_engine": st.session_state.source_engine,
                                "deployment": st.session_state.deployment_option,
                                "storage_type": st.session_state.storage_type,
                                "on_prem_cores": server_spec['cores'],
                                "peak_cpu_percent": server_spec['cpu_util'],
                                "on_prem_ram_gb": server_spec['ram'],
                                "peak_ram_percent": server_spec['ram_util'],
                                "storage_current_gb": server_spec['storage'],
                                "storage_growth_rate": server_spec.get('growth_rate', 20)/100,
                                "years": server_spec.get('years', 3),
                                "enable_encryption": server_spec.get('enable_encryption', True),
                                "enable_perf_insights": server_spec.get('enable_perf_insights', True),
                                "enable_enhanced_monitoring": server_spec.get('enable_enhanced_monitoring', False),
                                "monthly_data_transfer_gb": server_spec.get('monthly_transfer_gb', 100),
                                "max_iops": server_spec.get('max_iops', 1000),
                                "max_throughput_mbps": server_spec.get('max_throughput_mbps', 125),
                                "max_connections": server_spec.get('max_connections', 500)
                            }
                            
                            results = calculator.generate_comprehensive_recommendations(inputs)
                            st.session_state.results = results
                            st.session_state.generation_time = time.time() - start_time
                            
                            if calculator.ai_client:
                                with st.spinner("🤖 Generating AI insights..."):
                                    try:
                                        ai_insights = asyncio.run(calculator.generate_ai_insights(results, inputs))
                                        st.session_state.ai_insights = ai_insights
                                    except Exception as e:
                                        st.warning(f"AI insights generation failed for single server: {e}")
                                        st.session_state.ai_insights = None
                            
                            st.success(f"✅ Analysis complete in {st.session_state.generation_time:.1f} seconds!")
                            
                        except Exception as e:
                            st.error(f"❌ Error generating recommendations: {str(e)}")
                            st.code(traceback.format_exc())
            
            with col2:
                if st.button("🔄 Refresh AWS Pricing", use_container_width=True):
                    with st.spinner("Refreshing pricing data..."):
                        pricing_api.clear_cache()
                        st.success("✅ Pricing cache cleared!")
        
        else:
            # ================================
            # BULK SERVER ANALYSIS
            # ================================
            st.subheader("📊 Bulk Server Analysis")
            
            servers = st.session_state.on_prem_servers
            st.info(f"📋 Ready to analyze {len(servers)} servers")
            
            # Bulk Analysis Settings
            with st.expander("⚙️ Bulk Analysis Settings"):
                col1, col2 = st.columns(2)
                
                with col1:
                    batch_size = st.number_input("Batch Size", min_value=1, max_value=10, value=3, help="Number of servers to process simultaneously")
                    include_dev_environments = st.checkbox("Include DEV/TEST environments", value=False)
                
                with col2:
                    export_individual_reports = st.checkbox("Export individual server reports", value=False)
                    enable_parallel_processing = st.checkbox("Enable parallel processing", value=True)

            # BULK ANALYSIS BUTTON (SINGLE LOCATION)
            if st.button("🚀 Start Bulk Analysis", type="primary", use_container_width=True):
                # Check if calculator is properly initialized
                if not calculator:
                    st.error("❌ Calculator not initialized. Please configure migration settings first.")
                    st.stop()
                
                # Get servers from session state
                servers_to_analyze = st.session_state.on_prem_servers
                total_servers = len(servers_to_analyze)
                
                # Initialize progress tracking
                progress_bar = st.progress(0)
                status_placeholder = st.empty()
                results_placeholder = st.empty()
                
                bulk_results = {}
                total_monthly_cost_for_ai_insights = 0

                try:
                    for i, server in enumerate(servers_to_analyze):
                        # Update status for current server being analyzed
                        status_placeholder.text(f"🔄 Analyzing {server['server_name']} ({i+1}/{total_servers})")
                        
                        try:
                            inputs = {
                                "region": st.session_state.region,
                                "target_engine": st.session_state.target_engine,
                                "source_engine": server.get('database_engine', st.session_state.source_engine),
                                "deployment": st.session_state.deployment_option,
                                "storage_type": st.session_state.storage_type,
                                "on_prem_cores": server['cpu_cores'],
                                "peak_cpu_percent": server['peak_cpu_percent'],
                                "on_prem_ram_gb": server['ram_gb'],
                                "peak_ram_percent": server['peak_ram_percent'],
                                "storage_current_gb": server['storage_gb'],
                                "storage_growth_rate": 0.2,
                                "years": 3,
                                "enable_encryption": True,
                                "enable_perf_insights": True,
                                "enable_enhanced_monitoring": False,
                                "monthly_data_transfer_gb": 100,
                                "max_iops": server['max_iops'],
                                "max_throughput_mbps": server['max_throughput_mbps']
                            }
                            
                            server_results = calculator.generate_comprehensive_recommendations(inputs)
                            bulk_results[server['server_name']] = server_results
                            
                            # Accumulate cost for AI insights, only from successful analyses
                            if 'error' not in server_results:
                                prod_result = server_results.get('PROD', list(server_results.values())[0])
                                if 'error' not in prod_result:
                                    total_monthly_cost_for_ai_insights += safe_get(prod_result, 'total_cost', 0)

                            # Show success status
                            status_placeholder.success(f"✅ Completed {server['server_name']} ({i+1}/{total_servers})")

                        except Exception as e:
                            bulk_results[server['server_name']] = {'error': str(e)}
                            st.warning(f"⚠️ Error analyzing {server['server_name']}: {e}")
                            status_placeholder.error(f"❌ Failed {server['server_name']} ({i+1}/{total_servers})")
                        
                        # Update progress bar
                        progress_bar.progress((i + 1) / total_servers)
                        
                        # Update results summary
                        with results_placeholder.container():
                            completed_count = i + 1
                            successful_count = len([r for r in bulk_results.values() if 'error' not in r])
                            failed_count = completed_count - successful_count
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Completed", f"{completed_count}/{total_servers}")
                            with col2:
                                st.metric("Successful", successful_count)
                            with col3:
                                st.metric("Failed", failed_count)
                            
                            # Show current progress percentage
                            progress_pct = (completed_count / total_servers) * 100
                            st.progress(progress_pct / 100, text=f"Overall Progress: {progress_pct:.1f}%")
                        
                        # Small delay to make progress visible
                        time.sleep(0.5)
                    
                    # Store results
                    st.session_state.bulk_results = bulk_results
                    
                    successful_analyses = len([r for r in bulk_results.values() if 'error' not in r])
                    failed_analyses = total_servers - successful_analyses
                    
                    progress_bar.progress(1.0)
                    status_placeholder.success(f"🎉 Bulk analysis complete! {successful_analyses} successful, {failed_analyses} failed")
                    
                    # Clear the intermediate results display
                    results_placeholder.empty()
                    
                    # --- AI INSIGHTS GENERATION FOR BULK ANALYSIS ---
                    if successful_analyses > 0 and calculator.ai_client:
                        with st.spinner("🤖 Generating AI insights for bulk analysis..."):
                            try:
                                # Aggregate all successful results for AI analysis
                                aggregated_results_for_ai = {}
                                for server_name, server_data in bulk_results.items():
                                    if 'error' not in server_data:
                                        # Get the PROD environment result, or first available result
                                        if 'PROD' in server_data:
                                            aggregated_results_for_ai[server_name] = server_data['PROD']
                                        else:
                                            # Get first non-error result
                                            for env_key, env_result in server_data.items():
                                                if 'error' not in env_result:
                                                    aggregated_results_for_ai[server_name] = env_result
                                                    break

                                # Create input for the overall bulk AI insight
                                bulk_inputs_for_ai = {
                                    "region": st.session_state.region,
                                    "target_engine": st.session_state.target_engine,
                                    "source_engine": st.session_state.source_engine,
                                    "deployment": st.session_state.deployment_option,
                                    "storage_type": st.session_state.storage_type,
                                    "num_servers_analyzed": successful_analyses,
                                    "total_monthly_cost": total_monthly_cost_for_ai_insights,
                                    "analysis_mode": "bulk",
                                    "servers_summary": {
                                        "total_servers": len(st.session_state.on_prem_servers),
                                        "successful_analyses": successful_analyses,
                                        "failed_analyses": len(st.session_state.on_prem_servers) - successful_analyses
                                    }
                                }

                                # Calculate averages safely
                                if successful_analyses > 0:
                                    total_vcpus = 0
                                    total_ram = 0
                                    total_storage = 0
                                    
                                    for server_result in aggregated_results_for_ai.values():
                                        if 'writer' in server_result:
                                            total_vcpus += safe_get(server_result['writer'], 'actual_vCPUs', 0)
                                            total_ram += safe_get(server_result['writer'], 'actual_RAM_GB', 0)
                                        else:
                                            total_vcpus += safe_get(server_result, 'actual_vCPUs', 0)
                                            total_ram += safe_get(server_result, 'actual_RAM_GB', 0)
                                        
                                        total_storage += safe_get(server_result, 'storage_GB', 0)
                                    
                                    bulk_inputs_for_ai.update({
                                        "avg_cpu_cores": total_vcpus / successful_analyses,
                                        "avg_ram_gb": total_ram / successful_analyses,
                                        "avg_storage_gb": total_storage / successful_analyses
                                    })
                                
                                # Generate AI insights
                                bulk_ai_insights = asyncio.run(calculator.generate_ai_insights(aggregated_results_for_ai, bulk_inputs_for_ai))
                                st.session_state.ai_insights = bulk_ai_insights
                                st.success("✅ AI insights for bulk analysis generated!")
                                
                            except Exception as e:
                                st.warning(f"AI insights generation for bulk failed: {e}")
                                st.code(traceback.format_exc())
                                st.session_state.ai_insights = None
                                
                    elif successful_analyses == 0:
                        st.info("No successful server analyses found; skipping AI insights generation for bulk.")
                        st.session_state.ai_insights = None
                    elif not calculator.ai_client:
                        st.info("Anthropic API key not provided or AI client not ready; skipping AI insights generation.")
                        st.session_state.ai_insights = None

                    # Display results summary if successful
                    if successful_analyses > 0:
                        st.subheader("📊 Bulk Analysis Results") 
                        
                        summary_fig = create_bulk_analysis_summary_chart(bulk_results)
                        if summary_fig:
                            st.plotly_chart(summary_fig, use_container_width=True)
                        
                        summary_data = []
                        total_monthly_cost_display = 0
                        
                        for server_name, results in bulk_results.items():
                            if 'error' not in results:
                                result = results.get('PROD', list(results.values())[0])
                                if 'error' not in result:
                                    monthly_cost = safe_get(result, 'total_cost', 0)
                                    total_monthly_cost_display += monthly_cost
                                    
                                    recommended_instance_type = ""
                                    vcpus_display = 0
                                    ram_gb_display = 0
                                    if 'writer' in result:
                                        writer_info = result['writer']
                                        recommended_instance_type = safe_get_str(writer_info, 'instance_type', 'N/A')
                                        vcpus_display = safe_get(writer_info, 'actual_vCPUs', 0)
                                        ram_gb_display = safe_get(writer_info, 'actual_RAM_GB', 0)
                                        if result['readers']:
                                            recommended_instance_type += f" + {len(result['readers'])} Readers"
                                    else:
                                        recommended_instance_type = safe_get_str(result, 'instance_type', 'N/A')
                                        vcpus_display = safe_get(result, 'actual_vCPUs', 0)
                                        ram_gb_display = safe_get(result, 'actual_RAM_GB', 0)

                                    summary_data.append({
                                        'Server Name': server_name,
                                        'Recommended Instance': recommended_instance_type,
                                        'vCPUs': vcpus_display,
                                        'RAM (GB)': ram_gb_display,
                                        'Storage (GB)': safe_get(result, 'storage_GB', 0),
                                        'Monthly Cost': f"${monthly_cost:,.2f}",
                                        'Annual Cost': f"${monthly_cost * 12:,.2f}"
                                    })
                            else:
                                summary_data.append({
                                    'Server Name': server_name,
                                    'Recommended Instance': 'ERROR',
                                    'vCPUs': 0,
                                    'RAM (GB)': 0,
                                    'Storage (GB)': 0,
                                    'Monthly Cost': '$0.00',
                                    'Annual Cost': f"Error: {results['error']}"
                                })
                        
                        summary_df = pd.DataFrame(summary_data)
                        st.dataframe(summary_df, use_container_width=True)
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Monthly Cost", f"${total_monthly_cost_display:,.2f}")
                        with col2:
                            st.metric("Total Annual Cost", f"${total_monthly_cost_display * 12:,.2f}")
                        with col3:
                            avg_cost = total_monthly_cost_display / successful_analyses if successful_analyses > 0 else 0
                            st.metric("Average Cost per Server", f"${avg_cost:,.2f}")
                
                except Exception as e:
                    st.error(f"❌ Bulk analysis failed: {str(e)}")
                    st.code(traceback.format_exc())

        # ================================
        # DISPLAY RESULTS (BOTH SINGLE AND BULK)
        # ================================
        
        if st.session_state.results or st.session_state.bulk_results:
            current_results = st.session_state.results if st.session_state.current_analysis_mode == 'single' else st.session_state.bulk_results
            
            # Key metrics for single server
            if st.session_state.current_analysis_mode == 'single' and st.session_state.results:
                results = st.session_state.results
                
                st.subheader("📊 Key Metrics")
                
                valid_results = {k: v for k, v in results.items() if 'error' not in v}
                if valid_results:
                    prod_result = valid_results.get('PROD', list(valid_results.values())[0])
                    total_cost = safe_get(prod_result, 'total_cost', 0)
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        instance_type_display = ""
                        if 'writer' in prod_result:
                            writer_info = prod_result['writer']
                            instance_type_display = safe_get_str(writer_info, 'instance_type', 'N/A')
                            st.markdown(f"""
                            <div class="metric-container">
                                <div class="metric-value">{instance_type_display}</div>
                                <div class="metric-label">Writer Instance Type</div>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            instance_type_display = safe_get_str(prod_result, 'instance_type', 'N/A')
                            st.markdown(f"""
                            <div class="metric-container">
                                <div class="metric-value">{instance_type_display}</div>
                                <div class="metric-label">Production Instance</div>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    with col2:
                        vcpus_display = 0
                        ram_gb_display = 0
                        if 'writer' in prod_result:
                            writer_info = prod_result['writer']
                            vcpus_display = safe_get(writer_info, 'actual_vCPUs', 0)
                            ram_gb_display = safe_get(writer_info, 'actual_RAM_GB', 0)
                            st.markdown(f"""
                            <div class="metric-container">
                                <div class="metric-value">{vcpus_display} / {ram_gb_display}GB</div>
                                <div class="metric-label">Writer vCPUs / RAM</div>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            vcpus_display = safe_get(prod_result, 'actual_vCPUs', 0)
                            ram_gb_display = safe_get(prod_result, 'actual_RAM_GB', 0)
                            st.markdown(f"""
                            <div class="metric-container">
                                <div class="metric-value">{vcpus_display} / {ram_gb_display}GB</div>
                                <div class="metric-label">vCPUs / RAM</div>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    with col3:
                        st.markdown(f"""
                        <div class="metric-container">
                            <div class="metric-value">${total_cost:,.0f}</div>
                            <div class="metric-label">Total Monthly Cost</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col4:
                        migration_type_display = st.session_state.calculator.migration_profile.migration_type.value.title() if st.session_state.calculator.migration_profile else "N/A"
                        st.markdown(f"""
                        <div class="metric-container">
                            <div class="metric-value">{migration_type_display}</div>
                            <div class="metric-label">Migration Type</div>
                        </div>
                        """, unsafe_allow_html=True)

                    # Display Reader information if available
                    if 'readers' in prod_result and prod_result['readers']:
                        st.markdown("<br><h4>Reader Instances:</h4>", unsafe_allow_html=True)
                        for i, reader_info in enumerate(prod_result['readers']):
                            col_r1, col_r2, col_r3, col_r4 = st.columns(4)
                            with col_r1:
                                st.markdown(f"""
                                <div class="reader-box">
                                    <div class="metric-label">Reader {i+1} Type</div>
                                    <div class="metric-value" style="font-size:1.5rem;">{safe_get_str(reader_info, 'instance_type', 'N/A')}</div>
                                </div>
                                """, unsafe_allow_html=True)
                            with col_r2:
                                st.markdown(f"""
                                <div class="reader-box">
                                    <div class="metric-label">vCPUs</div>
                                    <div class="metric-value" style="font-size:1.5rem;">{safe_get(reader_info, 'actual_vCPUs', 0)}</div>
                                </div>
                                """, unsafe_allow_html=True)
                            with col_r3:
                                st.markdown(f"""
                                <div class="reader-box">
                                    <div class="metric-label">RAM (GB)</div>
                                    <div class="metric-value" style="font-size:1.5rem;">{safe_get(reader_info, 'actual_RAM_GB', 0)}</div>
                                </div>
                                """, unsafe_allow_html=True)
                            with col_r4:
                                st.markdown(f"""
                                <div class="reader-box">
                                    <div class="metric-label">Cost (Monthly)</div>
                                    <div class="metric-value" style="font-size:1.5rem;">${safe_get(reader_info, 'instance_cost', 0):,.0f}</div>
                                </div>
                                """, unsafe_allow_html=True)

    # ================================
    # DATA TRANSFER ANALYSIS SECTION
    # ================================
    
    st.subheader("🚛 Data Transfer Analysis")

    # Check if we have the necessary components
    if not transfer_calculator:
        st.warning("⚠️ Transfer calculator not initialized. Please check the data_transfer_calculator.py file.")
    elif st.session_state.current_analysis_mode == 'single':
        # Single Server Transfer Analysis
        if 'current_server_spec' in st.session_state:
            server_spec = st.session_state.current_server_spec
            data_size_gb = server_spec.get('storage', 500)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**🌐 Network Configuration**")
                dx_bandwidth = st.selectbox(
                    "Direct Connect Bandwidth",
                    ["1Gbps", "10Gbps", "100Gbps"],
                    index=1,
                    key="dx_bandwidth_select"
                )
                internet_bandwidth = st.number_input(
                    "Internet Bandwidth (Mbps)",
                    min_value=10,
                    max_value=10000,
                    value=100,
                    step=10,
                    key="internet_bandwidth_input"
                )
            
            with col2:
                st.markdown("**📊 Data Configuration**")
                include_transaction_logs = st.checkbox(
                    "Include Transaction Logs (+20%)",
                    value=True,
                    key="include_logs_checkbox"
                )
                compression_type = st.selectbox(
                    "Data Type",
                    ["database", "logs", "mixed", "none"],
                    index=0,
                    key="compression_type_select"
                )
            
            # Calculate effective data size
            effective_data_size = data_size_gb * 1.2 if include_transaction_logs else data_size_gb
            
            if st.button("🚀 Calculate Transfer Options", type="primary", use_container_width=True):
                with st.spinner("Calculating transfer options..."):
                    try:
                        # Store these values in session_state for potential use in reports
                        st.session_state.dx_bandwidth = dx_bandwidth
                        st.session_state.internet_bandwidth = internet_bandwidth
                        st.session_state.compression_type = compression_type
                        
                        # Convert DX bandwidth to Gbps
                        dx_gbps = float(dx_bandwidth.replace('Gbps', ''))
                        
                        # Calculate transfer analysis
                        transfer_results = transfer_calculator.calculate_comprehensive_transfer_analysis(
                            data_size_gb=effective_data_size,
                            region=st.session_state.region,
                            dx_bandwidth_gbps=dx_gbps,
                            internet_bandwidth_mbps=internet_bandwidth,
                            compression_type=compression_type
                        )
                        
                        # Store results
                        st.session_state.transfer_results = transfer_results
                        st.session_state.transfer_data_size = effective_data_size
                        
                        st.success("✅ Transfer analysis complete!")
                        
                    except Exception as e:
                        st.error(f"❌ Error calculating transfer options: {str(e)}")
            
            # Display results if available
            if hasattr(st.session_state, 'transfer_results') and st.session_state.transfer_results:
                transfer_results = st.session_state.transfer_results
                data_size = st.session_state.transfer_data_size
                
                st.subheader("📊 Transfer Results")
                
                # Key metrics
                col1, col2, col3, col4 = st.columns(4)
                
                # Find fastest and cheapest methods
                fastest_method = min(transfer_results.items(), key=lambda x: x[1].transfer_time_hours)
                cheapest_method = min(transfer_results.items(), key=lambda x: x[1].total_cost)
                
                with col1:
                    st.metric("Total Data Size", f"{data_size:,.0f} GB")
                
                with col2:
                    st.metric("Fastest Transfer", f"{fastest_method[1].transfer_time_days:.1f} days")
                
                with col3:
                    st.metric("Lowest Cost", f"${cheapest_method[1].total_cost:.0f}")
                
                with col4:
                    # Calculate time savings
                    dx_time = transfer_results.get('datasync_dx', fastest_method[1]).transfer_time_days
                    internet_time = transfer_results.get('datasync_internet', fastest_method[1]).transfer_time_days
                    time_saved = max(0, internet_time - dx_time)
                    st.metric("Time Saved (DX)", f"{time_saved:.1f} days")
                
                # Results table
                st.subheader("📋 Transfer Options Comparison")
                
                transfer_data = []
                for method, result in transfer_results.items():
                    method_name = result.recommended_method
                    transfer_data.append({
                        'Transfer Method': method_name,
                        'Transfer Time': f"{result.transfer_time_days:.1f} days ({result.transfer_time_hours:.1f} hours)",
                        'Total Cost': f"${result.total_cost:.2f}",
                        'Bandwidth Utilization': f"{result.bandwidth_utilization:.0f}%",
                        'Estimated Downtime': f"{result.estimated_downtime_hours:.1f} hours"
                    })
                
                transfer_df = pd.DataFrame(transfer_data)
                st.dataframe(transfer_df, use_container_width=True)
                
                # Simple cost breakdown
                st.subheader("💰 Cost Breakdown")
                
                for method, result in transfer_results.items():
                    with st.expander(f"{result.recommended_method} - ${result.total_cost:.2f}"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown(f"**Transfer Time:** {result.transfer_time_days:.1f} days")
                            st.markdown(f"**Estimated Downtime:** {result.estimated_downtime_hours:.1f} hours")
                        
                        with col2:
                            if result.cost_breakdown:
                                st.markdown("**Cost Components:**")
                                for component, cost in result.cost_breakdown.items():
                                    component_name = component.replace('_', ' ').title()
                                    st.markdown(f"- {component_name}: ${cost:.2f}")
                
                # Simple recommendations
                st.subheader("🎯 Recommendations")
                
                dx_result = transfer_results.get('datasync_dx')
                internet_result = transfer_results.get('datasync_internet')
                
                if dx_result and internet_result:
                    time_savings = internet_result.transfer_time_hours - dx_result.transfer_time_hours
                    cost_difference = dx_result.total_cost - internet_result.total_cost
                    
                    if time_savings > 24 and cost_difference < internet_result.total_cost * 0.5:  # Saves >1 day, <50% cost increase
                        recommendation = "✅ **Recommended: DataSync over Direct Connect**"
                        reasoning = f"Saves {time_savings/24:.1f} days for only ${abs(cost_difference):.0f} additional cost"
                    elif cost_difference < 0:  # DX is actually cheaper
                        recommendation = "✅ **Recommended: DataSync over Direct Connect**"
                        reasoning = "Faster AND cheaper than internet transfer"
                    else:
                        recommendation = "🌐 **Consider: DataSync over Internet**"
                        reasoning = f"More cost-effective option (saves ${abs(cost_difference):.0f})"
                    
                    st.markdown(f"""
                    <div class="advisory-box">
                        <h4>{recommendation}</h4>
                        <p><strong>Analysis:</strong> {reasoning}</p>
                        <p><strong>Data Size:</strong> {data_size:,.0f} GB (including logs: {'Yes' if include_transaction_logs else 'No'})</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Export option
                st.subheader("📊 Export Transfer Analysis")
                
                if st.button("📥 Export Transfer Results CSV", use_container_width=True):
                    csv_data = transfer_df.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv_data,
                        file_name=f"transfer_analysis_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
        else:
            st.info("💡 Configure server specifications in TAB 2 first to enable single server transfer analysis.")

    # Handle bulk analysis mode
    elif st.session_state.current_analysis_mode == 'bulk':
        if st.session_state.on_prem_servers:
            # Simple bulk transfer analysis
            st.info("🔄 Bulk transfer analysis: Calculate individual server transfers and sum the results")
            
            servers = st.session_state.on_prem_servers
            total_data_gb = sum([server['storage_gb'] for server in servers]) * 1.2  # Include logs (assuming 20% log overhead)
            
            col1, col2 = st.columns(2)
            
            with col1:
                bulk_dx_bandwidth = st.selectbox(
                    "Direct Connect Bandwidth",
                    ["1Gbps", "10Gbps", "100Gbps"],
                    index=1,
                    key="bulk_dx_bandwidth_select"
                )
            
            with col2:
                parallel_transfers = st.checkbox(
                    "Enable Parallel Transfers",
                    value=True,
                    key="parallel_transfers_checkbox"
                )
                if parallel_transfers:
                    max_concurrent = st.number_input(
                        "Max Concurrent Transfers",
                        min_value=1,
                        max_value=10,
                        value=5,
                        key="max_concurrent_transfers"
                    )
                else:
                    max_concurrent = 1
            
            if st.button("🚀 Calculate Bulk Transfer", type="primary", use_container_width=True):
                with st.spinner("Calculating bulk transfer options..."):
                    try:
                        bulk_dx_gbps = float(bulk_dx_bandwidth.replace('Gbps', ''))
                        total_cost_dx = 0
                        total_cost_internet = 0
                        max_time_dx = 0
                        max_time_internet = 0
                        
                        # Store individual transfer results for potential detailed view/PDF
                        individual_transfer_results = {}

                        for server in servers:
                            server_data_gb = server['storage_gb'] * 1.2
                            
                            # Calculate individual server transfer results
                            server_transfer_results = transfer_calculator.calculate_comprehensive_transfer_analysis(
                                data_size_gb=server_data_gb,
                                region=st.session_state.region,
                                dx_bandwidth_gbps=bulk_dx_gbps, 
                                internet_bandwidth_mbps=1000,
                                compression_type='database'
                            )
                            
                            # Accumulate costs and times
                            dx_result_single = server_transfer_results['datasync_dx']
                            internet_result_single = server_transfer_results['datasync_internet']
                            
                            total_cost_dx += dx_result_single.total_cost
                            total_cost_internet += internet_result_single.total_cost
                            
                            # For parallel: the total time is the time of the longest single transfer
                            # For sequential: sum up all times
                            if parallel_transfers:
                                max_time_dx = max(max_time_dx, dx_result_single.transfer_time_days)
                                max_time_internet = max(max_time_internet, internet_result_single.transfer_time_days)
                            else:
                                max_time_dx += dx_result_single.transfer_time_days
                                max_time_internet += internet_result_single.transfer_time_days
                            
                            individual_transfer_results[server['server_name']] = server_transfer_results

                        # If parallel, adjust total time by dividing by max_concurrent
                        if parallel_transfers:
                            max_time_dx /= max_concurrent
                            max_time_internet /= max_concurrent

                        # Store bulk transfer results for the overall view
                        st.session_state.transfer_results = {
                            'datasync_dx': TransferMethodResult(
                                recommended_method='AWS DataSync (Direct Connect)',
                                transfer_time_hours=max_time_dx * 24,
                                transfer_time_days=max_time_dx,
                                total_cost=total_cost_dx,
                                bandwidth_utilization=0,
                                estimated_downtime_hours=0,
                                cost_breakdown={}
                            ),
                             'datasync_internet': TransferMethodResult(
                                recommended_method='AWS DataSync (Internet)',
                                transfer_time_hours=max_time_internet * 24,
                                transfer_time_days=max_time_internet,
                                total_cost=total_cost_internet,
                                bandwidth_utilization=0,
                                estimated_downtime_hours=0,
                                cost_breakdown={}
                            )
                        }
                        st.session_state.transfer_data_size = total_data_gb
                        st.session_state.bulk_individual_transfer_results = individual_transfer_results
                        
                        # Display bulk results
                        st.subheader("📊 Bulk Transfer Results (Aggregated)")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Total Data", f"{total_data_gb:,.0f} GB")
                        
                        with col2:
                            st.metric("DX Transfer Time", f"{max_time_dx:.1f} days")
                        
                        with col3:
                            st.metric("DX Total Cost", f"${total_cost_dx:,.0f}")
                        
                        with col4:
                            savings = max(0, max_time_internet - max_time_dx)
                            st.metric("Time Saved (DX vs Internet)", f"{savings:.1f} days")
                        
                        # Comparison table
                        bulk_comparison = pd.DataFrame([
                            {
                                'Method': 'DataSync over Direct Connect',
                                'Transfer Time': f"{max_time_dx:.1f} days",
                                'Total Cost': f"${total_cost_dx:,.0f}",
                                'Servers': len(servers),
                                'Parallel': 'Yes' if parallel_transfers else 'No'
                            },
                            {
                                'Method': 'DataSync over Internet',
                                'Transfer Time': f"{max_time_internet:.1f} days", 
                                'Total Cost': f"${total_cost_internet:,.0f}",
                                'Servers': len(servers),
                                'Parallel': 'Yes' if parallel_transfers else 'No'
                            }
                        ])
                        
                        st.dataframe(bulk_comparison, use_container_width=True)
                        st.success("✅ Bulk transfer analysis complete!")
                        
                    except Exception as e:
                        st.error(f"❌ Error in bulk transfer calculation: {str(e)}")
                        st.code(traceback.format_exc())
        else:
            st.info("💡 Upload or add server specifications for bulk analysis first.")

    # If no transfer calculator available
    else:
        st.error("❌ Data transfer calculator not available. Please check that data_transfer_calculator.py is in the correct location.")
# ================================
# TAB 4: FINANCIAL ANALYSIS
# ================================

with tab4:
    st.header("💰 Financial Analysis & Advanced Visualizations")
    
    current_results = None
    if st.session_state.current_analysis_mode == 'single' and st.session_state.results:
        current_results = st.session_state.results
        analysis_title = "Single Server"
    elif st.session_state.current_analysis_mode == 'bulk' and st.session_state.bulk_results:
        current_results = st.session_state.bulk_results
        analysis_title = "Bulk Analysis"
    
    if not current_results:
        st.info("💡 Generate sizing recommendations first to enable financial analysis.")
    else:
        if st.session_state.current_analysis_mode == 'single':
            results = current_results
            valid_results = {k: v for k, v in results.items() if 'error' not in v}
            
            if valid_results:
                st.subheader(f"📊 {analysis_title} Financial Summary")
                
                prod_result = valid_results.get('PROD', list(valid_results.values())[0])
                total_cost = safe_get(prod_result, 'total_cost', 0)
                
                total_instance_cost = 0
                if 'writer' in prod_result:
                    total_instance_cost = safe_get(prod_result['cost_breakdown'], 'writer_monthly', 0) + \
                                          safe_get(prod_result['cost_breakdown'], 'readers_monthly', 0)
                else:
                    total_instance_cost = safe_get(prod_result, 'instance_cost', 0)
                
                total_storage_cost = safe_get(prod_result, 'storage_cost', 0)
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.markdown(f"""
                    <div class="metric-container">
                        <div class="metric-value">${total_cost:,.0f}</div>
                        <div class="metric-label">Total Monthly Cost</div>
                    </div>
                    """, unsafe_allow_html=True)

                with col2:
                    st.markdown(f"""
                    <div class="metric-container">
                        <div class="metric-value">${total_cost * 12:,.0f}</div>
                        <div class="metric-label">Annual Cost</div>
                    </div>
                    """, unsafe_allow_html=True)

                with col3:
                    instance_pct = (total_instance_cost/total_cost)*100 if total_cost > 0 else 0
                    st.markdown(f"""
                    <div class="metric-container">
                        <div class="metric-value">{instance_pct:.0f}%</div>
                        <div class="metric-label">Instance Cost %</div>
                    </div>
                    """, unsafe_allow_html=True)

                with col4:
                    storage_pct = (total_storage_cost/total_cost)*100 if total_cost > 0 else 0
                    st.markdown(f"""
                    <div class="metric-container">
                        <div class="metric-value">{storage_pct:.0f}%</div>
                        <div class="metric-label">Storage Cost %</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.subheader("📈 Advanced Financial Visualizations")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                    heatmap_fig = create_cost_heatmap(results)
                    if heatmap_fig:
                        st.plotly_chart(heatmap_fig, use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col2:
                    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                    if st.session_state.calculator and st.session_state.calculator.workload_characteristics:
                        workload_fig = create_workload_distribution_pie(st.session_state.calculator.workload_characteristics)
                        if workload_fig:
                            st.plotly_chart(workload_fig, use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                
        else:
            st.subheader(f"📊 {analysis_title} Financial Summary")
            
            total_servers = len(current_results)
            successful_servers = 0
            total_monthly_cost_display = 0
            total_annual_cost_display = 0
            server_costs = []
            
            for server_name, server_results in current_results.items():
                if 'error' not in server_results:
                    successful_servers += 1
                    result = server_results.get('PROD', list(server_results.values())[0])
                    if 'error' not in result:
                        monthly_cost = safe_get(result, 'total_cost', 0)
                        total_monthly_cost_display += monthly_cost
                        total_annual_cost_display += monthly_cost * 12
                        
                        instance_type_for_chart = ""
                        vcpus_for_chart = 0
                        ram_gb_for_chart = 0
                        if 'writer' in result:
                            writer_info = result['writer']
                            instance_type_for_chart = safe_get_str(writer_info, 'instance_type', 'Unknown')
                            vcpus_for_chart = safe_get(writer_info, 'actual_vCPUs', 0)
                            ram_gb_for_chart = safe_get(writer_info, 'actual_RAM_GB', 0)
                        else:
                            instance_type_for_chart = safe_get_str(result, 'instance_type', 'Unknown')
                            vcpus_for_chart = safe_get(result, 'actual_vCPUs', 0)
                            ram_gb_for_chart = safe_get(result, 'actual_RAM_GB', 0)

                        server_costs.append({
                            'Server Name': server_name,
                            'Monthly Cost': monthly_cost,
                            'Instance Type': instance_type_for_chart,
                            'vCPUs': vcpus_for_chart,
                            'RAM (GB)': ram_gb_for_chart
                        })
            
            if successful_servers > 0:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(f"""
                    <div class="metric-container">
                        <div class="metric-value">${total_monthly_cost_display:,.0f}</div>
                        <div class="metric-label">Total Monthly Cost (All Servers)</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col2:
                    st.markdown(f"""
                    <div class="metric-container">
                        <div class="metric-value">${total_annual_cost_display:,.0f}</div>
                        <div class="metric-label">Total Annual Cost (All Servers)</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col3:
                    avg_cost_per_server = total_monthly_cost_display / successful_servers
                    st.markdown(f"""
                    <div class="metric-container">
                        <div class="metric-value">${avg_cost_per_server:,.0f}</div>
                        <div class="metric-label">Avg. Monthly Cost per Server</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.subheader("📈 Bulk Analysis Visualizations")
                bulk_summary_fig = create_bulk_analysis_summary_chart(current_results)
                if bulk_summary_fig:
                    st.plotly_chart(bulk_summary_fig, use_container_width=True, key='bulk_chart_unique')
            else:
                st.info("No successful bulk analysis results to display.")

    # Transfer Cost Analysis
    if hasattr(st.session_state, 'transfer_results') and st.session_state.transfer_results:
        st.subheader("🚛 Data Transfer Cost Analysis")
        
        transfer_results = st.session_state.transfer_results
        
        # Transfer cost summary
        total_transfer_costs = sum([result.total_cost for result in transfer_results.values()])
        min_transfer_cost = min([result.total_cost for result in transfer_results.values()])
        max_transfer_cost = max([result.total_cost for result in transfer_results.values()])
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-value">${min_transfer_cost:.0f}</div>
                <div class="metric-label">Minimum Transfer Cost</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-value">${max_transfer_cost:.0f}</div>
                <div class="metric-label">Maximum Transfer Cost</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            cost_range = max_transfer_cost - min_transfer_cost
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-value">${cost_range:.0f}</div>
                <div class="metric-label">Cost Range</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Compare transfer costs with infrastructure costs
        if st.session_state.current_analysis_mode == 'single' and st.session_state.results:
            valid_results = {k: v for k, v in st.session_state.results.items() if 'error' not in v}
            if valid_results:
                prod_result = valid_results.get('PROD', list(valid_results.values())[0])
                monthly_infrastructure_cost = safe_get(prod_result, 'total_cost', 0)
                
                st.markdown("**💡 Cost Comparison: Transfer vs Infrastructure**")
                
                # Create comparison chart
                comparison_data = {
                    'Category': ['Monthly Infrastructure', 'One-time Transfer (Min)', 'One-time Transfer (Max)'],
                    'Cost': [monthly_infrastructure_cost, min_transfer_cost, max_transfer_cost],
                    'Type': ['Recurring', 'One-time', 'One-time']
                }
                
                fig = px.bar(
                    comparison_data,
                    x='Category',
                    y='Cost',
                    color='Type',
                    title='💰 Transfer Cost vs Infrastructure Cost Comparison',
                    text='Cost'
                )
                
                fig.update_traces(texttemplate='$%{text:,.0f}', textposition='outside')
                fig.update_layout(height=400)
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Calculate cost as percentage of infrastructure
                min_percentage = (min_transfer_cost / monthly_infrastructure_cost) * 100 if monthly_infrastructure_cost > 0 else 0
                max_percentage = (max_transfer_cost / monthly_infrastructure_cost) * 100 if monthly_infrastructure_cost > 0 else 0
                
                st.markdown(f"""
                <div class="advisory-box">
                    <strong>Transfer Cost Analysis:</strong><br>
                    • Minimum transfer cost represents {min_percentage:.1f}% of monthly infrastructure cost<br>
                    • Maximum transfer cost represents {max_percentage:.1f}% of monthly infrastructure cost<br>
                    • Transfer is a one-time cost vs recurring monthly infrastructure costs
                </div>
                """, unsafe_allow_html=True)

# ================================
# TAB 5: AI INSIGHTS
# ================================

with tab5:
    st.header("🤖 AI Insights & Recommendations")
    
    # Check for AI insights availability
    if not st.session_state.ai_insights:
        st.info("💡 Generate sizing recommendations first to enable AI insights.")
        
        # Show what's needed for AI insights
        st.subheader("🔧 AI Insights Requirements")
        
        col1, col2 = st.columns(2)
        
        with col1:
            api_key_status = "✅ Available" if (st.session_state.user_claude_api_key_input or 
                                               ("anthropic" in st.secrets and "ANTHROPIC_API_KEY" in st.secrets["anthropic"]) or
                                               os.environ.get('ANTHROPIC_API_KEY')) else "❌ Missing"
            st.write(f"**Anthropic API Key:** {api_key_status}")
            
            analysis_status = "✅ Complete" if (st.session_state.results or st.session_state.bulk_results) else "❌ Pending"
            st.write(f"**Analysis Results:** {analysis_status}")
        
        with col2:
            mode = st.session_state.current_analysis_mode.title()
            st.write(f"**Analysis Mode:** {mode}")
            
            if st.session_state.current_analysis_mode == 'single':
                server_spec_status = "✅ Available" if 'current_server_spec' in st.session_state else "❌ Missing"
                st.write(f"**Server Specs:** {server_spec_status}")
            else:
                server_count = len(st.session_state.on_prem_servers) if st.session_state.on_prem_servers else 0
                st.write(f"**Bulk Servers:** {server_count} configured")
        
        if api_key_status == "❌ Missing":
            st.warning("⚠️ Please provide your Anthropic API key at the top of the page to enable AI insights.")
    
    else:
        ai_insights = st.session_state.ai_insights
        
        # Check for errors in AI insights
        if isinstance(ai_insights, dict) and ai_insights.get("error"):
            st.error(f"❌ Error retrieving AI insights: {ai_insights['error']}")
            
            # Offer to retry AI insights generation
            if st.button("🔄 Retry AI Insights Generation", type="primary"):
                with st.spinner("🤖 Regenerating AI insights..."):
                    try:
                        calculator = st.session_state.calculator
                        
                        if st.session_state.current_analysis_mode == 'single' and st.session_state.results:
                            # Retry single server AI insights
                            server_spec = st.session_state.current_server_spec
                            inputs = {
                                "region": st.session_state.region,
                                "target_engine": st.session_state.target_engine,
                                "source_engine": st.session_state.source_engine,
                                "deployment": st.session_state.deployment_option,
                                "storage_type": st.session_state.storage_type,
                                "on_prem_cores": server_spec['cores'],
                                "peak_cpu_percent": server_spec['cpu_util'],
                                "on_prem_ram_gb": server_spec['ram'],
                                "peak_ram_percent": server_spec['ram_util']
                            }
                            
                            ai_insights = asyncio.run(calculator.generate_ai_insights(st.session_state.results, inputs))
                            st.session_state.ai_insights = ai_insights
                            st.success("✅ AI insights regenerated successfully!")
                            st.rerun()
                            
                        elif st.session_state.current_analysis_mode == 'bulk' and st.session_state.bulk_results:
                            # Retry bulk AI insights using the fixed logic
                            successful_results = {k: v for k, v in st.session_state.bulk_results.items() if 'error' not in v}
                            
                            if successful_results:
                                aggregated_results = {}
                                for server_name, server_data in successful_results.items():
                                    if 'PROD' in server_data:
                                        aggregated_results[server_name] = server_data['PROD']
                                    else:
                                        for env_key, env_result in server_data.items():
                                            if 'error' not in env_result:
                                                aggregated_results[server_name] = env_result
                                                break
                                
                                bulk_inputs = {
                                    "region": st.session_state.region,
                                    "target_engine": st.session_state.target_engine,
                                    "source_engine": st.session_state.source_engine,
                                    "deployment": st.session_state.deployment_option,
                                    "storage_type": st.session_state.storage_type,
                                    "num_servers_analyzed": len(successful_results),
                                    "analysis_mode": "bulk"
                                }
                                
                                ai_insights = asyncio.run(calculator.generate_ai_insights(aggregated_results, bulk_inputs))
                                st.session_state.ai_insights = ai_insights
                                st.success("✅ Bulk AI insights regenerated successfully!")
                                st.rerun()
                        
                    except Exception as e:
                        st.error(f"❌ Failed to regenerate AI insights: {e}")
                        
        else:
            # Display AI insights successfully
            st.markdown("""
            <div class="ai-insight-card">
                <h3>🤖 AI-Powered Analysis from Claude</h3>
                <p>Leveraging advanced AI to provide deeper insights into your migration.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Show analysis type
            analysis_type = st.session_state.current_analysis_mode.title()
            st.subheader(f"📊 {analysis_type} Analysis AI Insights")
            
            # Key metrics row
            col1, col2, col3 = st.columns(3)
            
            with col1:
                risk_level = ai_insights.get('risk_level', 'N/A')
                st.markdown(f"""
                <div class="metric-container">
                    <div class="metric-value">{risk_level}</div>
                    <div class="metric-label">Migration Risk Level</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                cost_opt_potential = ai_insights.get('cost_optimization_potential', 0)
                if isinstance(cost_opt_potential, (int, float)):
                    cost_opt_display = f"{cost_opt_potential * 100:.0f}%"
                else:
                    cost_opt_display = "N/A"
                    
                st.markdown(f"""
                <div class="metric-container">
                    <div class="metric-value">{cost_opt_display}</div>
                    <div class="metric-label">Cost Optimization Potential</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                writers = ai_insights.get('recommended_writers', 'N/A')
                readers = ai_insights.get('recommended_readers', 'N/A')
                
                if writers != 'N/A' and readers != 'N/A':
                    arch_display = f"{writers}W / {readers}R"
                else:
                    arch_display = "Standard RDS"
                    
                st.markdown(f"""
                <div class="metric-container">
                    <div class="metric-value">{arch_display}</div>
                    <div class="metric-label">AI Recommended Arch.</div>
                </div>
                """, unsafe_allow_html=True)

            # Comprehensive AI Analysis
            st.subheader("🔍 Comprehensive AI Analysis")
            
            ai_analysis_text = ai_insights.get("ai_analysis", "No detailed AI analysis available.")
            
            if ai_analysis_text and ai_analysis_text != "No detailed AI analysis available.":
                st.markdown('<div class="advisory-box">', unsafe_allow_html=True)
                
                # Split long AI analysis into paragraphs for better readability
                if len(ai_analysis_text) > 1000:
                    paragraphs = ai_analysis_text.split('\n\n')
                    for paragraph in paragraphs:
                        if paragraph.strip():
                            st.write(paragraph.strip())
                else:
                    st.write(ai_analysis_text)
                    
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.info("AI analysis is generating. Please wait or refresh the page.")
            
            # Migration Phases (if available)
            st.subheader("📅 Recommended Migration Phases")
            
            migration_phases = ai_insights.get("recommended_migration_phases")
            if migration_phases and isinstance(migration_phases, list):
                st.markdown('<div class="phase-timeline">', unsafe_allow_html=True)
                for i, phase in enumerate(migration_phases):
                    st.markdown(f"**Phase {i+1}:** {phase}")
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                # Provide default migration phases
                default_phases = [
                    "Assessment & Discovery (2-3 weeks)",
                    "Schema Conversion & Testing (3-4 weeks)", 
                    "Data Migration Setup (1-2 weeks)",
                    "Application Code Conversion (4-6 weeks)",
                    "User Acceptance Testing (2-3 weeks)",
                    "Production Cutover (1 week)",
                    "Post-Migration Optimization (2-4 weeks)"
                ]
                
                st.markdown('<div class="phase-timeline">', unsafe_allow_html=True)
                st.info("Using standard migration phases (AI-specific phases not available)")
                for i, phase in enumerate(default_phases):
                    st.markdown(f"**Phase {i+1}:** {phase}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Additional insights for bulk analysis
            if st.session_state.current_analysis_mode == 'bulk':
                st.subheader("📊 Bulk Analysis Specific Insights")
                
                total_servers = len(st.session_state.bulk_results) if st.session_state.bulk_results else 0
                successful_servers = len([r for r in st.session_state.bulk_results.values() if 'error' not in r]) if st.session_state.bulk_results else 0
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Servers Analyzed", total_servers)
                
                with col2:
                    st.metric("Successful Analyses", successful_servers)
                
                with col3:
                    success_rate = (successful_servers / total_servers * 100) if total_servers > 0 else 0
                    st.metric("Success Rate", f"{success_rate:.1f}%")
                
                if successful_servers < total_servers:
                    failed_servers = total_servers - successful_servers
                    st.warning(f"⚠️ {failed_servers} servers failed analysis. Check server specifications and try again.")
            
            # Export AI insights
            st.subheader("📄 Export AI Insights")
            
            if st.button("📥 Export AI Insights as Text", use_container_width=True):
                insights_export = f"""# AWS RDS Migration AI Insights
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Analysis Mode: {st.session_state.current_analysis_mode.title()}

## Risk Assessment
Migration Risk Level: {ai_insights.get('risk_level', 'N/A')}
Cost Optimization Potential: {cost_opt_display}

## Recommended Architecture
{arch_display}

## Comprehensive Analysis
{ai_analysis_text}

## Migration Phases
"""
                
                if migration_phases:
                    for i, phase in enumerate(migration_phases):
                        insights_export += f"{i+1}. {phase}\n"
                else:
                    insights_export += "Standard migration phases recommended\n"
                
                st.download_button(
                    label="📥 Download AI Insights",
                    data=insights_export,
                    file_name=f"ai_insights_{st.session_state.current_analysis_mode}_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                    mime="text/plain",
                    use_container_width=True
                )

# ================================
# TAB 6: REPORTS
# ================================

with tab6:
    st.header("📋 PDF Report Generator")

    current_analysis_results = None
    current_server_specs_for_pdf = None

    if st.session_state.current_analysis_mode == 'single' and st.session_state.results:
        current_analysis_results = st.session_state.results
        current_server_specs_for_pdf = st.session_state.get('current_server_spec')
        analysis_mode_for_pdf = 'single'
    elif st.session_state.current_analysis_mode == 'bulk' and st.session_state.bulk_results:
        current_analysis_results = st.session_state.bulk_results
        current_server_specs_for_pdf = st.session_state.get('on_prem_servers')
        analysis_mode_for_pdf = 'bulk'
    
    # Enhanced PDF Generation with better error handling
    if current_analysis_results:
        # Show current analysis summary
        st.subheader("📊 Current Analysis Summary")
        if analysis_mode_for_pdf == 'single':
            st.info(f"✅ Single server analysis ready for PDF generation")
            if 'current_server_spec' in st.session_state:
                st.write(f"**Server:** {st.session_state.current_server_spec.get('server_name', 'Unknown')}")
        else:
            successful_count = len([r for r in current_analysis_results.values() if 'error' not in r])
            st.info(f"✅ Bulk analysis ready: {successful_count} successful servers")
        
        # AI Insights Status
        if st.session_state.ai_insights:
            st.success("🤖 AI insights available and will be included")
        else:
            st.warning("⚠️ No AI insights available - PDF will be generated without AI analysis")
        
        # Transfer Results Status
        if hasattr(st.session_state, 'transfer_results') and st.session_state.transfer_results:
            st.success("🚛 Transfer analysis available and will be included")
        else:
            st.info("💡 No transfer analysis - PDF will be generated without transfer section")
        
        # PDF Generation Options
        st.subheader("⚙️ PDF Generation Options")
        col1, col2 = st.columns(2)
        
        with col1:
            include_detailed_specs = st.checkbox("Include Detailed Server Specifications", value=True)
            include_cost_breakdown = st.checkbox("Include Cost Breakdown", value=True)
        
        with col2:
            include_migration_timeline = st.checkbox("Include Migration Timeline", value=True)
            include_risk_assessment = st.checkbox("Include Risk Assessment", value=True)
        
        # Generate PDF Button with enhanced error handling
        if st.button("📄 Generate Enhanced PDF Report", type="primary", use_container_width=True):
            with st.spinner("Generating Enhanced PDF Report..."):
                try:
                    # Validate inputs before PDF generation
                    validation_errors = []
                    
                    if not current_analysis_results:
                        validation_errors.append("No analysis results available")
                    
                    if analysis_mode_for_pdf == 'single' and not current_server_specs_for_pdf:
                        validation_errors.append("Server specifications missing for single analysis")
                    
                    if analysis_mode_for_pdf == 'bulk' and not current_server_specs_for_pdf:
                        validation_errors.append("Server specifications missing for bulk analysis")
                    
                    if validation_errors:
                        st.error("❌ Validation errors:")
                        for error in validation_errors:
                            st.write(f"• {error}")
                        st.stop()
                    
                    # Enhanced PDF generation with better error handling
                    st.info("🔄 Initializing Enhanced Report Generator...")
                    
                    # Create enhanced generator instance
                    comprehensive_generator = ImprovedReportGenerator()
                    
                    st.info("🔄 Preparing report data...")
                    
                    # Prepare AI insights (ensure it's not None)
                    ai_insights_for_pdf = st.session_state.ai_insights if st.session_state.ai_insights else {
                        "risk_level": "Unknown",
                        "cost_optimization_potential": 0,
                        "ai_analysis": "AI insights were not available during analysis generation."
                    }
                    
                    # Prepare transfer results
                    transfer_results_for_pdf = None
                    if hasattr(st.session_state, 'transfer_results') and st.session_state.transfer_results:
                        transfer_results_for_pdf = st.session_state.transfer_results
                    
                    st.info("🔄 Generating comprehensive PDF...")
                    
                    # Generate PDF with comprehensive error handling
                    # Generate comprehensive PDF with charts and detailed analysis
                    pdf_bytes = generate_improved_pdf_report(
                    analysis_results=current_analysis_results,
                    analysis_mode=analysis_mode_for_pdf,
                    server_specs=current_server_specs_for_pdf,
                    ai_insights=ai_insights_for_pdf,
                    transfer_results=transfer_results_for_pdf

                    )
                    
                    if pdf_bytes:
                        st.success("✅ Enhanced PDF Report generated successfully!")
                        
                        # Calculate file size
                        file_size_mb = len(pdf_bytes) / (1024 * 1024)
                        st.info(f"📄 PDF Size: {file_size_mb:.2f} MB")
                        
                        # Generate filename with timestamp
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                        filename = f"aws_rds_migration_enhanced_{analysis_mode_for_pdf}_{timestamp}.pdf"
                        
                        st.download_button(
                            label="📥 Download Enhanced PDF Report",
                            data=pdf_bytes,
                            file_name=filename,
                            mime="application/pdf",
                            use_container_width=True
                        )
                        
                        # Show report contents summary
                        st.subheader("📋 Report Contents")
                        contents = [
                            "✅ Executive Summary",
                            "✅ Migration Strategy & Planning",
                            "✅ Technical Specifications",
                            "✅ Financial Analysis",
                            "✅ Risk Assessment"
                        ]
                        
                        if ai_insights_for_pdf and 'ai_analysis' in ai_insights_for_pdf:
                            contents.append("✅ AI-Powered Insights")
                        
                        if transfer_results_for_pdf:
                            contents.append("✅ Data Transfer Analysis")
                        
                        for content in contents:
                            st.write(content)
                    
                    else:
                        st.error("❌ Failed to generate PDF. The report generator returned None.")
                        st.info("💡 This could be due to:")
                        st.write("• Missing reportlab dependencies")
                        st.write("• Memory issues with large datasets")
                        st.write("• Data formatting problems")
                        
                        # Offer alternative export
                        st.subheader("🔄 Alternative Export Options")
                        
                        # JSON export as fallback
                        if st.button("📊 Export Analysis as JSON (Fallback)", use_container_width=True):
                            export_data = {
                                'analysis_mode': analysis_mode_for_pdf,
                                'analysis_results': current_analysis_results,
                                'server_specs': current_server_specs_for_pdf,
                                'ai_insights': ai_insights_for_pdf,
                                'transfer_results': transfer_results_for_pdf,
                                'generated_at': datetime.now().isoformat()
                            }
                            
                            json_data = json.dumps(export_data, indent=2, default=str)
                            
                            st.download_button(
                                label="📥 Download JSON Report",
                                data=json_data,
                                file_name=f"analysis_report_{timestamp}.json",
                                mime="application/json",
                                use_container_width=True
                            )
                
                except ImportError as e:
                    st.error(f"❌ Missing required libraries for PDF generation: {e}")
                    st.info("💡 Please ensure required libraries are installed:")
                    st.code("pip install reportlab kaleido plotly>=5.0.0")
                
                except Exception as e:
                    st.error(f"❌ Error generating enhanced PDF report: {str(e)}")
                    st.code(traceback.format_exc())
                    
                    # Show detailed error information
                    st.subheader("🔍 Debug Information")
                    st.write("**Analysis Mode:**", analysis_mode_for_pdf)
                    st.write("**Has Results:**", bool(current_analysis_results))
                    st.write("**Has Specs:**", bool(current_server_specs_for_pdf))
                    st.write("**Has AI Insights:**", bool(st.session_state.ai_insights))
                    st.write("**Has Transfer Results:**", bool(hasattr(st.session_state, 'transfer_results') and st.session_state.transfer_results))
    
    else:
        st.warning("⚠️ Please run an analysis first (Single or Bulk) before generating the PDF report.")
        
        # Show what's needed
        st.subheader("📋 Prerequisites for PDF Generation")
        st.write("1. Configure migration settings in **Migration Planning** tab")
        st.write("2. Set up server specifications in **Server Specifications** tab")
        st.write("3. Run analysis in **Sizing Analysis** tab")
        st.write("4. Optionally generate AI insights and transfer analysis")
        st.write("5. Return to this tab to generate comprehensive PDF report")

    # Additional export options
    st.subheader("📊 Additional Export Options")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**📈 Executive Summary**")
        if st.button("Generate Executive Summary", use_container_width=True):
            if st.session_state.current_analysis_mode == 'single' and st.session_state.results:
                valid_results = {k: v for k, v in st.session_state.results.items() if 'error' not in v}
                if valid_results:
                    prod_result = valid_results.get('PROD', list(valid_results.values())[0])
                    
                    exec_summary = f"""
# Executive Summary - AWS RDS Migration Analysis

## Migration Overview
- **Source Engine:** {st.session_state.source_engine}
- **Target Engine:** {st.session_state.target_engine}
- **Migration Type:** {st.session_state.calculator.migration_profile.migration_type.value.title() if st.session_state.calculator.migration_profile else 'Unknown'}

## Cost Analysis
- **Monthly Cost:** ${safe_get(prod_result, 'total_cost', 0):,.2f}
- **Annual Cost:** ${safe_get(prod_result, 'total_cost', 0) * 12:,.2f}

## Recommended Configuration
"""
                    if 'writer' in prod_result:
                        writer_info = prod_result['writer']
                        exec_summary += f"- **Writer Instance:** {safe_get_str(writer_info, 'instance_type', 'N/A')}\n"
                        exec_summary += f"- **Writer Resources:** {safe_get(writer_info, 'actual_vCPUs', 0)} vCPUs, {safe_get(writer_info, 'actual_RAM_GB', 0)} GB RAM\n"
                        if prod_result['readers']:
                            exec_summary += f"- **Reader Instances:** {len(prod_result['readers'])} x {safe_get_str(prod_result['readers'][0], 'instance_type', 'N/A')}\n"
                    else:
                        exec_summary += f"- **Instance Type:** {safe_get_str(prod_result, 'instance_type', 'N/A')}\n"
                        exec_summary += f"- **Resources:** {safe_get(prod_result, 'actual_vCPUs', 0)} vCPUs, {safe_get(prod_result, 'actual_RAM_GB', 0)} GB RAM\n"
                    
                    exec_summary += f"- **Storage:** {safe_get(prod_result, 'storage_GB', 0)} GB\n"
                    
                    if st.session_state.ai_insights and 'ai_analysis' in st.session_state.ai_insights:
                        exec_summary += f"\n## AI Recommendations\n{st.session_state.ai_insights['ai_analysis'][:500]}...\n"
                    
                    st.download_button(
                        label="📥 Download Executive Summary",
                        data=exec_summary,
                        file_name=f"executive_summary_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
                        mime="text/markdown",
                        use_container_width=True
                    )
            elif st.session_state.current_analysis_mode == 'bulk' and st.session_state.bulk_results:
                # Generate Executive Summary for Bulk Analysis
                total_servers_summary = len(st.session_state.bulk_results)
                successful_servers_summary = sum(1 for result in st.session_state.bulk_results.values() if 'error' not in result)
                total_monthly_cost_summary = sum(safe_get(result.get('PROD', {}), 'total_cost', 0) for result in st.session_state.bulk_results.values() if 'error' not in result)
                
                exec_summary_bulk = f"""
# Executive Summary - AWS RDS Bulk Migration Analysis

## Migration Overview
- **Source Engine:** {st.session_state.source_engine}
- **Target Engine:** {st.session_state.target_engine}
- **Migration Type:** {st.session_state.calculator.migration_profile.migration_type.value.title() if st.session_state.calculator.migration_profile else 'Unknown'}

## Aggregate Cost Analysis
- **Total Servers Analyzed:** {total_servers_summary}
- **Successful Analyses:** {successful_servers_summary}
- **Total Monthly Cost (Aggregated):** ${total_monthly_cost_summary:,.2f}
- **Total Annual Cost (Aggregated):** ${total_monthly_cost_summary * 12:,.2f}

## Key AI Insights (Overall Migration)
"""
                if st.session_state.ai_insights and 'ai_analysis' in st.session_state.ai_insights:
                    exec_summary_bulk += f"{st.session_state.ai_insights['ai_analysis'][:500]}...\n"
                else:
                    exec_summary_bulk += "No AI insights available for the overall bulk migration.\n"
                
                st.download_button(
                    label="📥 Download Executive Summary",
                    data=exec_summary_bulk,
                    file_name=f"executive_summary_bulk_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
                    mime="text/markdown",
                    use_container_width=True
                )
            else:
                st.info("Executive summary available after running an analysis.")
    
    with col2:
        st.markdown("**📋 Technical Specifications**")
        if st.button("Export Technical Specs", use_container_width=True):
            if st.session_state.current_analysis_mode == 'single' and 'current_server_spec' in st.session_state:
                tech_specs = {
                    'server_specification': st.session_state.current_server_spec,
                    'migration_config': {
                        'source_engine': st.session_state.source_engine,
                        'target_engine': st.session_state.target_engine,
                        'deployment_option': st.session_state.deployment_option,
                        'region': st.session_state.region,
                        'storage_type': st.session_state.storage_type
                    },
                    'recommendations': st.session_state.results,
                    'generated_at': datetime.now().isoformat()
                }
                
                tech_specs_json = json.dumps(tech_specs, indent=2, default=str)
                
                st.download_button(
                    label="📥 Download Tech Specs",
                    data=tech_specs_json,
                    file_name=f"technical_specifications_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                    mime="application/json",
                    use_container_width=True
                )
            elif st.session_state.current_analysis_mode == 'bulk' and st.session_state.bulk_results:
                bulk_tech_specs = {
                    'analysis_mode': 'bulk',
                    'migration_config': {
                        'source_engine': st.session_state.source_engine,
                        'target_engine': st.session_state.target_engine,
                        'deployment_option': st.session_state.deployment_option,
                        'region': st.session_state.region,
                        'storage_type': st.session_state.storage_type
                    },
                    'bulk_servers_specifications': st.session_state.on_prem_servers,
                    'bulk_recommendations': st.session_state.bulk_results,
                    'generated_at': datetime.now().isoformat()
                }
                bulk_tech_specs_json = json.dumps(bulk_tech_specs, indent=2, default=str)
                st.download_button(
                    label="📥 Download Bulk Tech Specs",
                    data=bulk_tech_specs_json,
                    file_name=f"technical_specifications_bulk_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                    mime="application/json",
                    use_container_width=True
                )
            else:
                st.info("Technical specifications available after running an analysis.")
    
    with col3:
        st.markdown("**💰 Cost Analysis Report**")
        if st.button("Generate Cost Report", use_container_width=True):
            cost_analysis = {
                'analysis_mode': st.session_state.current_analysis_mode,
                'migration_type': st.session_state.calculator.migration_profile.migration_type.value if st.session_state.calculator.migration_profile else 'unknown',
                'cost_summary': {},
                'generated_at': datetime.now().isoformat()
            }
            
            if st.session_state.current_analysis_mode == 'single' and st.session_state.results:
                valid_results = {k: v for k, v in st.session_state.results.items() if 'error' not in v}
                cost_analysis['cost_summary'] = {
                    'environments': {},
                    'total_monthly': 0,
                    'total_annual': 0
                }
                
                for env, result in valid_results.items():
                    monthly_cost = safe_get(result, 'total_cost', 0)
                    cost_analysis['cost_summary']['environments'][env] = {
                        'monthly_cost': monthly_cost,
                        'annual_cost': monthly_cost * 12,
                        'cost_breakdown': safe_get(result, 'cost_breakdown', {})
                    }
                    cost_analysis['cost_summary']['total_monthly'] += monthly_cost
                
                cost_analysis['cost_summary']['total_annual'] = cost_analysis['cost_summary']['total_monthly'] * 12
            
            elif st.session_state.current_analysis_mode == 'bulk' and st.session_state.bulk_results:
                cost_analysis['cost_summary'] = {
                    'servers': {},
                    'total_monthly': 0,
                    'total_annual': 0,
                    'average_monthly_per_server': 0
                }
                
                successful_servers = 0
                
                for server_name, server_results in st.session_state.bulk_results.items():
                    if 'error' not in server_results:
                        result = server_results.get('PROD', list(server_results.values())[0])
                        if 'error' not in result:
                            monthly_cost = safe_get(result, 'total_cost', 0)
                            cost_analysis['cost_summary']['servers'][server_name] = {
                                'monthly_cost': monthly_cost,
                                'annual_cost': monthly_cost * 12
                            }
                            cost_analysis['cost_summary']['total_monthly'] += monthly_cost
                            successful_servers += 1
                
                cost_analysis['cost_summary']['total_annual'] = cost_analysis['cost_summary']['total_monthly'] * 12
                cost_analysis['cost_summary']['average_monthly_per_server'] = cost_analysis['cost_summary']['total_monthly'] / max(successful_servers, 1)
            
            cost_report_json = json.dumps(cost_analysis, indent=2, default=str)
            
            st.download_button(
                label="📥 Download Cost Report",
                data=cost_report_json,
                file_name=f"cost_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                mime="application/json",
                use_container_width=True
            )

# ================================
# FOOTER
# ================================

st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <h5>🚀 Enterprise AWS RDS Migration & Sizing Tool v2.0</h5>
    <p>AI-Powered Database Migration Analysis • Built for Enterprise Scale</p>
    <p>💡 For support and advanced features, contact your AWS solutions architect</p>
</div>
""", unsafe_allow_html=True)
                   