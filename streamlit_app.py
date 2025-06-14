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
from data_transfer_calculator import DataTransferCalculator, TransferMethod
import io
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch



# Add to imports section
try:
    from enhanced_report_generator import EnhancedReportGenerator, StreamlitReportIntegration
except ImportError as e:
    st.error(f"Enhanced report generator not available: {e}")
    # Fall back to existing report generator
# Authentication imports
import bcrypt
import jwt

# Import our enhanced modules
try:
    from rds_sizing import EnhancedRDSSizingCalculator, MigrationType, WorkloadCharacteristics
    from aws_pricing import EnhancedAWSPricingAPI
    from report_generator import EnhancedReportGenerator
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
                        st.session_state.user_email = user_data['email']
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
            st.session_state.user_email = token_data['email']
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
# In streamlit_app.py, add this class after the utility functions section (around line 500-600)
# Add after the existing utility functions and before the main app initialization

class StreamlitEnhancedReportGenerator:
    """Enhanced report generator integrated with Streamlit app"""
    
    def __init__(self):
        self.setup_report_styles()
    
    def setup_report_styles(self):
        """Setup report styles"""
        # Report setup code here
        pass
    
    def generate_bulk_report_in_chunks(self, servers_list, chunk_size=10):
        """Generate bulk reports in chunks to manage memory"""
        total_chunks = len(servers_list) // chunk_size + (1 if len(servers_list) % chunk_size else 0)
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        processed_results = []
        for i in range(0, len(servers_list), chunk_size):
            chunk = servers_list[i:i+chunk_size]
            
            # Update progress
            progress = (i + chunk_size) / len(servers_list)
            progress_bar.progress(min(progress, 1.0))
            status_text.text(f"Processing servers {i+1}-{min(i+chunk_size, len(servers_list))} of {len(servers_list)}")
            
            # Process chunk
            chunk_result = self._process_server_chunk(chunk)
            processed_results.append(chunk_result)
            
            # Explicit garbage collection
            import gc
            gc.collect()
        
        progress_bar.progress(1.0)
        status_text.text("Report generation complete!")
        
        return processed_results
    
    def _process_server_chunk(self, server_chunk):
        """Process a chunk of servers for bulk analysis"""
        chunk_data = []
        for server in server_chunk:
            # Use existing calculator to process individual server
            try:
                # Process using existing logic from streamlit app
                server_analysis = self._analyze_server_for_report(server)
                chunk_data.append(server_analysis)
            except Exception as e:
                st.warning(f"Error processing server {server.get('server_name', 'Unknown')}: {e}")
                chunk_data.append({'error': str(e), 'server_name': server.get('server_name', 'Unknown')})
        
        return chunk_data
    
    def _analyze_server_for_report(self, server):
        """Analyze individual server using existing calculator"""
        # This would use the existing calculator logic
        # You can reference st.session_state.calculator here
        if st.session_state.calculator:
            # Use existing calculation logic
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
            
            return st.session_state.calculator.generate_comprehensive_recommendations(inputs)
        
        return {'error': 'Calculator not available'}

# Initialize the enhanced report generator
if 'enhanced_report_generator' not in st.session_state:
    st.session_state.enhanced_report_generator = StreamlitEnhancedReportGenerator()


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

def create_transfer_time_comparison_chart(transfer_results):
    """Create transfer time comparison chart"""
    if not transfer_results:
        return None
    
    methods = []
    hours = []
    costs = []
    
    for method, result in transfer_results.items():
        methods.append(result.recommended_method)
        hours.append(result.transfer_time_hours)
        costs.append(result.total_cost)
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Transfer Time Comparison', 'Transfer Cost Comparison'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Transfer time chart
    fig.add_trace(
        go.Bar(
            x=methods,
            y=hours,
            name='Transfer Time (Hours)',
            marker_color='#1f77b4',
            text=[f'{h:.1f}h' for h in hours],
            textposition='auto'
        ),
        row=1, col=1
    )
    
    # Transfer cost chart
    fig.add_trace(
        go.Bar(
            x=methods,
            y=costs,
            name='Transfer Cost ($)',
            marker_color='#ff7f0e',
            text=[f'${c:.2f}' for c in costs],
            textposition='auto'
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        title_text="🚛 Data Transfer Analysis Comparison",
        showlegend=False,
        height=400
    )
    
    return fig

def create_transfer_timeline_chart(transfer_results, data_size_gb):
    """Create transfer timeline visualization"""
    if not transfer_results:
        return None
    
    fig = go.Figure()
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for i, (method, result) in enumerate(transfer_results.items()):
        fig.add_trace(go.Bar(
            x=[result.transfer_time_hours],
            y=[f"{result.recommended_method}"],
            orientation='h',
            name=f'{method} - {result.transfer_time_days:.1f} days',
            marker_color=colors[i % len(colors)],
            text=f'{result.transfer_time_days:.1f} days (${result.total_cost:.2f})',
            textposition='auto'
        ))
    
    fig.update_layout(
        title=f'📅 Transfer Timeline for {data_size_gb:,.0f}GB Data',
        xaxis_title='Time (Hours)',
        yaxis_title='Transfer Method',
        height=300,
        barmode='group'
    )
    
    return fig

def create_cost_breakdown_pie(transfer_result):
    """Create cost breakdown pie chart for selected transfer method"""
    if not transfer_result or not transfer_result.cost_breakdown:
        return None
    
    labels = list(transfer_result.cost_breakdown.keys())
    values = list(transfer_result.cost_breakdown.values())
    
    clean_labels = [label.replace('_', ' ').title() for label in labels]
    
    fig = go.Figure(data=[go.Pie(
        labels=clean_labels,
        values=values,
        hole=.3,
        marker_colors=['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
    )])
    
    fig.update_layout(
        title=f'💰 Cost Breakdown - {transfer_result.recommended_method}',
        height=300
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
# ================================
# ENHANCED REPORT GENERATOR
# ================================
# Add this section after the visualization functions (around line 700)
# and before the initialization section (before "# Initialize session state")

import io
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch

class EnhancedReportGenerator:
    """Enhanced PDF Report Generator for AWS RDS Migration Tool"""
    
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self.setup_custom_styles()
    
    def setup_custom_styles(self):
        """Setup custom styles for the report"""
        # Title style
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Title'],
            fontSize=24,
            spaceAfter=30,
            textColor=colors.darkblue,
            alignment=1  # Center alignment
        ))
        
        # Section header style
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading1'],
            fontSize=16,
            spaceBefore=20,
            spaceAfter=12,
            textColor=colors.darkblue,
            borderWidth=2,
            borderColor=colors.lightblue,
            borderPadding=5
        ))
        
        # Subsection header style
        self.styles.add(ParagraphStyle(
            name='SubsectionHeader',
            parent=self.styles['Heading2'],
            fontSize=14,
            spaceBefore=15,
            spaceAfter=8,
            textColor=colors.darkgreen
        ))
        
        # Highlight style for important information
        self.styles.add(ParagraphStyle(
            name='Highlight',
            parent=self.styles['Normal'],
            fontSize=12,
            textColor=colors.darkred,
            backColor=colors.lightyellow,
            borderWidth=1,
            borderColor=colors.orange,
            borderPadding=5
        ))

    def generate_comprehensive_pdf_report(self, analysis_results, analysis_mode, server_specs=None, ai_insights=None, transfer_results=None):
        """Generate a comprehensive PDF report for both single and bulk analysis"""
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4, 
                              rightMargin=0.5*inch, leftMargin=0.5*inch,
                              topMargin=1*inch, bottomMargin=1*inch)
        
        story = []
        
        # Title Page
        story.extend(self._create_title_page(analysis_mode))
        story.append(PageBreak())
        
        # Executive Summary
        story.extend(self._create_executive_summary(analysis_results, analysis_mode, ai_insights))
        story.append(PageBreak())
        
        # Migration Strategy Section
        story.extend(self._create_migration_strategy_section(analysis_results, ai_insights))
        story.append(PageBreak())
        
        if analysis_mode == 'single':
            # Single Server Analysis
            story.extend(self._create_single_server_analysis(analysis_results, server_specs))
        else:
            # Bulk Server Analysis
            story.extend(self._create_bulk_server_analysis(analysis_results, server_specs))
        
        story.append(PageBreak())
        
        # Financial Analysis
        story.extend(self._create_financial_analysis_section(analysis_results, analysis_mode))
        story.append(PageBreak())
        
        # Data Transfer Analysis (if available)
        if transfer_results:
            story.extend(self._create_transfer_analysis_section(transfer_results))
            story.append(PageBreak())
        
        # AI Insights Section
        if ai_insights:
            story.extend(self._create_ai_insights_section(ai_insights))
            story.append(PageBreak())
        
        # Risk Assessment & Implementation
        story.extend(self._create_risk_assessment_section(analysis_results, ai_insights))
        
        # Build the PDF
        try:
            doc.build(story)
            buffer.seek(0)
            return buffer.getvalue()
        except Exception as e:
            st.error(f"Error building PDF: {e}")
            return None

    def _create_title_page(self, analysis_mode):
        """Create an enhanced title page"""
        story = []
        
        # Main title
        title_text = f"AWS RDS Migration & Sizing Report"
        story.append(Paragraph(title_text, self.styles['CustomTitle']))
        story.append(Spacer(1, 0.5*inch))
        
        # Analysis type
        analysis_type = "Single Server Analysis" if analysis_mode == 'single' else "Bulk Server Analysis"
        story.append(Paragraph(f"<b>{analysis_type}</b>", self.styles['Heading1']))
        story.append(Spacer(1, 0.3*inch))
        
        # Generation details
        generation_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        story.append(Paragraph(f"<b>Generated:</b> {generation_time}", self.styles['Normal']))
        story.append(Paragraph(f"<b>Report Type:</b> Comprehensive Analysis & Recommendations", self.styles['Normal']))
        story.append(Paragraph(f"<b>Prepared for:</b> Enterprise Cloud Migration Team", self.styles['Normal']))
        
        story.append(Spacer(1, 1*inch))
        
        # Key highlights box
        highlights = [
            "✓ AI-Powered Sizing Recommendations",
            "✓ Cost Optimization Analysis", 
            "✓ Migration Risk Assessment",
            "✓ Performance Optimization Strategy",
            "✓ Implementation Roadmap"
        ]
        
        highlights_text = "<br/>".join(highlights)
        story.append(Paragraph(f"<b>Report Includes:</b><br/>{highlights_text}", self.styles['Highlight']))
        
        return story

    def _create_executive_summary(self, analysis_results, analysis_mode, ai_insights):
        """Create comprehensive executive summary"""
        story = []
        story.append(Paragraph("Executive Summary", self.styles['SectionHeader']))
        
        if analysis_mode == 'single':
            # Single server executive summary
            valid_results = {k: v for k, v in analysis_results.items() if 'error' not in v}
            if valid_results:
                prod_result = valid_results.get('PROD', list(valid_results.values())[0])
                total_monthly_cost = safe_get(prod_result, 'total_cost', 0)
                
                # Key metrics table
                summary_data = [
                    ['Metric', 'Value', 'Impact'],
                    ['Monthly Cost', f'${total_monthly_cost:,.2f}', 'Baseline operational cost'],
                    ['Annual Cost', f'${total_monthly_cost * 12:,.2f}', 'Total yearly investment'],
                    ['Migration Type', 'Heterogeneous', 'Requires conversion planning'],
                    ['Estimated Timeline', '3-4 months', 'Including testing & validation']
                ]
                
                table = Table(summary_data, colWidths=[2*inch, 1.5*inch, 2.5*inch])
                table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 12),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                story.append(table)
                story.append(Spacer(1, 20))
                
        else:
            # Bulk analysis executive summary
            total_servers = len(analysis_results)
            successful_servers = sum(1 for result in analysis_results.values() if 'error' not in result)
            total_monthly_cost = 0
            
            for server_results in analysis_results.values():
                if 'error' not in server_results:
                    result = server_results.get('PROD', list(server_results.values())[0])
                    if 'error' not in result:
                        total_monthly_cost += safe_get(result, 'total_cost', 0)
            
            # Bulk summary table
            bulk_summary_data = [
                ['Metric', 'Value', 'Analysis'],
                ['Total Servers', str(total_servers), f'{successful_servers} successful analyses'],
                ['Total Monthly Cost', f'${total_monthly_cost:,.2f}', 'All servers combined'],
                ['Average Cost per Server', f'${total_monthly_cost/max(successful_servers,1):,.2f}', 'Cost distribution'],
                ['Total Annual Cost', f'${total_monthly_cost * 12:,.2f}', 'Yearly investment'],
                ['Migration Complexity', 'Mixed', 'Varies by server configuration']
            ]
            
            table = Table(bulk_summary_data, colWidths=[2*inch, 1.5*inch, 2.5*inch])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.darkgreen),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.lightgreen),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(table)
            story.append(Spacer(1, 20))
        
        # AI insights summary
        if ai_insights:
            story.append(Paragraph("AI-Powered Key Insights", self.styles['SubsectionHeader']))
            
            risk_level = ai_insights.get('risk_level', 'UNKNOWN')
            cost_optimization = ai_insights.get('cost_optimization_potential', 0) * 100
            
            ai_summary = f"""
            <b>Migration Risk Assessment:</b> {risk_level}<br/>
            <b>Cost Optimization Potential:</b> {cost_optimization:.1f}%<br/>
            <b>Recommended Architecture:</b> {ai_insights.get('recommended_writers', 1)} Writer(s), {ai_insights.get('recommended_readers', 1)} Reader(s)<br/>
            <b>Success Probability:</b> High with proper planning and execution
            """
            
            story.append(Paragraph(ai_summary, self.styles['Highlight']))
        
        return story

    def _create_migration_strategy_section(self, analysis_results, ai_insights):
        """Create detailed migration strategy section"""
        story = []
        story.append(Paragraph("Migration Strategy & Planning", self.styles['SectionHeader']))
        
        # Migration approach
        story.append(Paragraph("Migration Approach", self.styles['SubsectionHeader']))
        
        approach_text = """
        This migration follows a comprehensive, phased approach designed to minimize risk and ensure business continuity.
        The strategy incorporates industry best practices and leverages AWS native services for optimal results.
        """
        story.append(Paragraph(approach_text, self.styles['Normal']))
        story.append(Spacer(1, 12))
        
        # Migration phases table
        phases_data = [['Phase', 'Duration', 'Key Activities', 'Success Criteria']]
        
        phase_details = [
            ('Assessment & Planning', '2-3 weeks', 'Schema analysis, workload assessment', 'Complete inventory & migration plan'),
            ('Schema Conversion', '3-4 weeks', 'AWS SCT, manual refactoring', '100% schema compatibility'),
            ('DMS Setup', '1-2 weeks', 'Replication instances, tasks', 'Successful initial sync'),
            ('Testing & Validation', '4-5 weeks', 'Functional & performance testing', 'All tests passing'),
            ('Cutover & Go-Live', '1 week', 'Final sync, DNS switch', 'Production operational'),
            ('Optimization', '2-3 weeks', 'Performance tuning, monitoring', 'SLA compliance achieved')
        ]
        
        for phase, duration, activities, criteria in phase_details:
            phases_data.append([phase, duration, activities, criteria])
        
        phases_table = Table(phases_data, colWidths=[1.5*inch, 1*inch, 2*inch, 1.5*inch])
        phases_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.navy),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.lightblue),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('VALIGN', (0, 0), (-1, -1), 'TOP')
        ]))
        story.append(phases_table)
        
        return story

    def _create_single_server_analysis(self, analysis_results, server_specs):
        """Create detailed single server analysis section"""
        story = []
        story.append(Paragraph("Single Server Analysis", self.styles['SectionHeader']))
        
        valid_results = {k: v for k, v in analysis_results.items() if 'error' not in v}
        
        for env, result in valid_results.items():
            story.append(Paragraph(f"{env} Environment Configuration", self.styles['SubsectionHeader']))
            
            # Instance configuration table
            config_data = [['Component', 'Specification', 'Performance Impact']]
            
            if 'writer' in result:
                # Aurora cluster configuration
                writer = result['writer']
                config_data.extend([
                    ['Writer Instance', safe_get_str(writer, 'instance_type', 'N/A'), 'Primary database operations'],
                    ['Writer vCPUs', str(safe_get(writer, 'actual_vCPUs', 'N/A')), 'Compute capacity'],
                    ['Writer RAM', f"{safe_get(writer, 'actual_RAM_GB', 'N/A')} GB", 'Memory for caching'],
                    ['Storage', f"{safe_get(result, 'storage_GB', 'N/A')} GB", 'Data and index storage']
                ])
                
                if result.get('readers'):
                    for i, reader in enumerate(result['readers']):
                        config_data.append([f'Reader {i+1}', safe_get_str(reader, 'instance_type', 'N/A'), 'Read scaling & availability'])
            else:
                # Standard RDS configuration
                config_data.extend([
                    ['Instance Type', safe_get_str(result, 'instance_type', 'N/A'), 'Compute and memory'],
                    ['vCPUs', str(safe_get(result, 'actual_vCPUs', 'N/A')), 'Processing power'],
                    ['RAM', f"{safe_get(result, 'actual_RAM_GB', 'N/A')} GB", 'Database caching'],
                    ['Storage', f"{safe_get(result, 'storage_GB', 'N/A')} GB", 'Data storage capacity']
                ])
            
            config_table = Table(config_data, colWidths=[1.5*inch, 2*inch, 2.5*inch])
            config_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 11),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.lightblue),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(config_table)
            story.append(Spacer(1, 15))
        
        return story

    def _create_bulk_server_analysis(self, analysis_results, server_specs):
        """Create comprehensive bulk server analysis"""
        story = []
        story.append(Paragraph("Bulk Server Analysis", self.styles['SectionHeader']))
        
        # Summary statistics
        total_servers = len(analysis_results)
        successful_analyses = sum(1 for result in analysis_results.values() if 'error' not in result)
        failed_analyses = total_servers - successful_analyses
        
        story.append(Paragraph(f"Analysis Summary: {successful_analyses} successful, {failed_analyses} failed out of {total_servers} total servers", self.styles['Normal']))
        story.append(Spacer(1, 12))
        
        # Aggregate cost analysis
        story.append(Paragraph("Cost Analysis by Server", self.styles['SubsectionHeader']))
        
        bulk_data = [['Server Name', 'Instance Type', 'Monthly Cost', 'Annual Cost', 'vCPUs', 'RAM (GB)']]
        total_monthly_cost = 0
        
        for server_name, server_results in analysis_results.items():
            if 'error' not in server_results:
                result = server_results.get('PROD', list(server_results.values())[0])
                if 'error' not in result:
                    monthly_cost = safe_get(result, 'total_cost', 0)
                    total_monthly_cost += monthly_cost
                    
                    instance_type = 'N/A'
                    vcpus = 0
                    ram_gb = 0
                    
                    if 'writer' in result:
                        writer = result['writer']
                        instance_type = safe_get_str(writer, 'instance_type', 'N/A')
                        vcpus = safe_get(writer, 'actual_vCPUs', 0)
                        ram_gb = safe_get(writer, 'actual_RAM_GB', 0)
                        if result.get('readers'):
                            instance_type += f" + {len(result['readers'])} readers"
                    else:
                        instance_type = safe_get_str(result, 'instance_type', 'N/A')
                        vcpus = safe_get(result, 'actual_vCPUs', 0)
                        ram_gb = safe_get(result, 'actual_RAM_GB', 0)
                    
                    bulk_data.append([
                        server_name[:15],  # Truncate long names
                        instance_type[:20],  # Truncate long instance types
                        f'${monthly_cost:.2f}',
                        f'${monthly_cost * 12:.2f}',
                        str(vcpus),
                        str(ram_gb)
                    ])
            else:
                bulk_data.append([server_name[:15], 'ERROR', '$0.00', '$0.00', '0', '0'])
        
        # Add totals row
        avg_monthly_cost = total_monthly_cost / max(successful_analyses, 1)
        bulk_data.append([
            'TOTALS/AVERAGES',
            f'{successful_analyses} servers',
            f'${total_monthly_cost:.2f}',
            f'${total_monthly_cost * 12:.2f}',
            f'Avg: ${avg_monthly_cost:.2f}',
            ''
        ])
        
        bulk_table = Table(bulk_data, colWidths=[1.2*inch, 1.3*inch, 1*inch, 1*inch, 0.7*inch, 0.8*inch])
        bulk_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.purple),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 9),
            ('FONTSIZE', (0, 1), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -2), colors.lavender),
            ('BACKGROUND', (0, -1), (-1, -1), colors.yellow),
            ('FONTNAME', (0, -1), (-1, -1), 'Helvetica-Bold'),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(bulk_table)
        
        return story

    def _create_financial_analysis_section(self, analysis_results, analysis_mode):
        """Create detailed financial analysis section"""
        story = []
        story.append(Paragraph("Financial Analysis & Cost Optimization", self.styles['SectionHeader']))
        
        # TCO Analysis
        story.append(Paragraph("Total Cost of Ownership (TCO) Analysis", self.styles['SubsectionHeader']))
        
        if analysis_mode == 'single':
            valid_results = {k: v for k, v in analysis_results.items() if 'error' not in v}
            if valid_results:
                prod_result = valid_results.get('PROD', list(valid_results.values())[0])
                monthly_cost = safe_get(prod_result, 'total_cost', 0)
        else:
            # Bulk TCO analysis
            monthly_cost = 0
            for server_results in analysis_results.values():
                if 'error' not in server_results:
                    result = server_results.get('PROD', list(server_results.values())[0])
                    if 'error' not in result:
                        monthly_cost += safe_get(result, 'total_cost', 0)
        
        # 3-year TCO projection
        tco_data = [['Year', 'AWS Costs', 'OpEx Savings', 'Net Position']]
        
        for year in range(1, 4):
            annual_aws_cost = monthly_cost * 12 * (1.03 ** (year - 1))  # 3% inflation
            opex_savings = (200000 if analysis_mode == 'bulk' else 150000) + (year * 30000)
            net_position = annual_aws_cost - opex_savings
            
            tco_data.append([
                f'Year {year}',
                f'${annual_aws_cost:,.0f}',
                f'${opex_savings:,.0f}',
                f'${net_position:,.0f}'
            ])
        
        tco_table = Table(tco_data, colWidths=[1*inch, 1.5*inch, 1.5*inch, 1.5*inch])
        tco_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkred),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.mistyrose),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(tco_table)
        
        return story

    def _create_transfer_analysis_section(self, transfer_results):
        """Create data transfer analysis section"""
        story = []
        story.append(Paragraph("Data Transfer Analysis", self.styles['SectionHeader']))
        
        # Transfer options comparison
        transfer_data = [['Method', 'Transfer Time', 'Cost', 'Recommended Use']]
        
        for method, result in transfer_results.items():
            transfer_data.append([
                result.recommended_method,
                f'{result.transfer_time_days:.1f} days',
                f'${result.total_cost:.2f}',
                'Time-critical' if result.transfer_time_hours < 24 else 'Cost-effective'
            ])
        
        transfer_table = Table(transfer_data, colWidths=[1.5*inch, 1.5*inch, 1*inch, 2*inch])
        transfer_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.orange),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.wheat),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(transfer_table)
        
        return story

    def _create_ai_insights_section(self, ai_insights):
        """Create AI insights section"""
        story = []
        story.append(Paragraph("AI-Powered Insights & Recommendations", self.styles['SectionHeader']))
        
        # AI analysis summary
        if 'ai_analysis' in ai_insights:
            story.append(Paragraph("Comprehensive AI Analysis", self.styles['SubsectionHeader']))
            ai_text = ai_insights['ai_analysis'][:2000] + "..." if len(ai_insights['ai_analysis']) > 2000 else ai_insights['ai_analysis']
            
            # Split long AI text into paragraphs
            ai_paragraphs = ai_text.split('. ')
            for i in range(0, len(ai_paragraphs), 3):  # Group every 3 sentences
                paragraph_text = '. '.join(ai_paragraphs[i:i+3])
                if paragraph_text.strip():
                    story.append(Paragraph(paragraph_text + ".", self.styles['Normal']))
                    story.append(Spacer(1, 10))
        
        return story

    def _create_risk_assessment_section(self, analysis_results, ai_insights):
        """Create risk assessment and mitigation section"""
        story = []
        story.append(Paragraph("Risk Assessment & Implementation Roadmap", self.styles['SectionHeader']))
        
        # Risk matrix
        risk_data = [
            ['Risk Category', 'Probability', 'Impact', 'Mitigation Strategy'],
            ['Schema Conversion', 'Medium', 'High', 'AWS SCT + Expert Review'],
            ['Performance Issues', 'Low', 'Medium', 'Load Testing + Tuning'],
            ['Data Corruption', 'Low', 'Critical', 'Validation Scripts + Checksums'],
            ['Extended Downtime', 'Medium', 'High', 'Parallel Sync + Quick Cutover'],
            ['Cost Overrun', 'Medium', 'Medium', 'Reserved Instances + Monitoring']
        ]
        
        risk_table = Table(risk_data, colWidths=[1.5*inch, 1.2*inch, 1.2*inch, 2.1*inch])
        risk_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkred),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.mistyrose),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('VALIGN', (0, 0), (-1, -1), 'TOP')
        ]))
        story.append(risk_table)
        
        # Implementation phases
        story.append(Spacer(1, 20))
        story.append(Paragraph("Implementation Phases", self.styles['SubsectionHeader']))
        
        phases_text = """
        <b>Phase 1:</b> Assessment & Schema Analysis (2-4 weeks)<br/>
        <b>Phase 2:</b> AWS SCT Schema Conversion (1-2 weeks)<br/>
        <b>Phase 3:</b> DMS Setup & Initial Data Migration (1-2 weeks)<br/>
        <b>Phase 4:</b> Application Code Conversion (4-8 weeks)<br/>
        <b>Phase 5:</b> Testing & Validation (2-4 weeks)<br/>
        <b>Phase 6:</b> Cutover & Go-Live (1 week)<br/>
        <b>Phase 7:</b> Post-Migration Optimization (2-4 weeks)
        """
        
        story.append(Paragraph(phases_text, self.styles['Highlight']))
        
        return story

    def generate_bulk_report_in_chunks(self, servers_list, chunk_size=10):
        """Generate bulk reports in chunks to manage memory"""
        total_chunks = len(servers_list) // chunk_size + (1 if len(servers_list) % chunk_size else 0)
        
        processed_results = {}
        
        for i in range(0, len(servers_list), chunk_size):
            chunk = servers_list[i:i+chunk_size]
            
            # Process chunk using existing Streamlit calculator
            for server in chunk:
                try:
                    if st.session_state.calculator:
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
                        
                        server_results = st.session_state.calculator.generate_comprehensive_recommendations(inputs)
                        processed_results[server['server_name']] = server_results
                    else:
                        processed_results[server['server_name']] = {'error': 'Calculator not available'}
                        
                except Exception as e:
                    processed_results[server['server_name']] = {'error': str(e)}
            
            # Explicit garbage collection
            import gc
            gc.collect()
        
        return processed_results

# ================================
# STREAMLIT INTEGRATION HELPER
# ================================

def generate_enhanced_pdf_report(analysis_results, analysis_mode, server_specs=None, ai_insights=None, transfer_results=None):
    """Helper function to generate enhanced PDF report in Streamlit"""
    try:
        enhanced_generator = EnhancedReportGenerator()
        
        pdf_bytes = enhanced_generator.generate_comprehensive_pdf_report(
            analysis_results=analysis_results,
            analysis_mode=analysis_mode,
            server_specs=server_specs,
            ai_insights=ai_insights,
            transfer_results=transfer_results
        )
        
        return pdf_bytes
        
    except Exception as e:
        st.error(f"Error generating enhanced PDF report: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        return None

# ================================
# END OF ENHANCED REPORT GENERATOR
# ================================


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


# Initialize Firebase
if st.session_state.firebase_app is None:
    st.session_state.firebase_app, st.session_state.firebase_auth, st.session_state.firebase_db = initialize_firebase()

# Initialize components
@st.cache_resource
def initialize_components():
    """Initialize the pricing API and report generator"""
    try:
        pricing_api = EnhancedAWSPricingAPI()
        report_generator = EnhancedReportGenerator()
        return pricing_api, report_generator
    except Exception as e:
        st.error(f"Error initializing static components: {e}")
        return None, None

pricing_api, report_generator = initialize_components()
if not pricing_api or not report_generator:
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

# Main navigation
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
                if st.session_state.user_claude_api_key_input and st.session_state.calculator.ai_client is None:
                    st.session_state.calculator = EnhancedRDSSizingCalculator(
                        anthropic_api_key=st.session_state.user_claude_api_key_input,
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
# TAB 2: SERVER SPECIFICATIONS
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
                col1, col2 = st.columns(2)
                with col1:
                    server_name = st.text_input("Server Name/Hostname", value="PROD-DB-01", key="server_name_input")
                    environment = st.selectbox("Environment", ["PROD", "UAT", "DEV", "TEST"], key="environment_select")
                with col2:
                    database_version = st.text_input("Database Version", value="12.1.0.2", key="db_version_input")
                    database_size_gb = st.number_input("Database Size (GB)", min_value=1, value=500, key="db_size_input")
            
            st.subheader("💾 Hardware Specifications")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown('<div class="spec-section">', unsafe_allow_html=True)
                st.markdown("**🖥️ Compute Resources**")
                cores = st.number_input("CPU Cores", min_value=1, max_value=128, value=16, step=1, key="cores_input")
                cpu_model = st.text_input("CPU Model", value="Intel Xeon E5-2690 v4", key="cpu_model_input")
                cpu_ghz = st.number_input("CPU Speed (GHz)", min_value=1.0, max_value=5.0, value=2.6, step=0.1, key="cpu_ghz_input")
                st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown('<div class="spec-section">', unsafe_allow_html=True)
                st.markdown("**💾 Memory Resources**")
                ram = st.number_input("Total RAM (GB)", min_value=1, max_value=1024, value=64, step=1, key="ram_input")
                ram_type = st.selectbox("RAM Type", ["DDR3", "DDR4", "DDR5"], index=1, key="ram_type_select")
                ram_speed = st.number_input("RAM Speed (MHz)", min_value=800, max_value=5000, value=2400, step=100, key="ram_speed_input")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="spec-section">', unsafe_allow_html=True)
                st.markdown("**💿 Storage Resources**")
                storage = st.number_input("Current Storage (GB)", min_value=10, value=500, step=10, key="storage_input")
                storage_type_hw = st.selectbox("Storage Type", ["HDD", "SSD", "NVMe SSD", "SAN"], index=2, key="storage_type_hw_select")
                raid_level = st.selectbox("RAID Level", ["RAID 0", "RAID 1", "RAID 5", "RAID 10", "RAID 6"], index=3, key="raid_level_select")
                st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown('<div class="spec-section">', unsafe_allow_html=True)
                st.markdown("**📈 Growth & Planning**")
                growth_rate = st.number_input("Annual Growth Rate (%)", min_value=0, max_value=100, value=20, key="growth_rate_input")
                years = st.number_input("Planning Horizon (years)", min_value=1, max_value=5, value=3, key="years_input")
                st.markdown('</div>', unsafe_allow_html=True)
            
            st.subheader("⚡ Performance Specifications")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown('<div class="spec-section">', unsafe_allow_html=True)
                st.markdown("**📊 CPU Performance**")
                cpu_util = st.slider("Peak CPU Utilization (%)", 1, 100, 75, key="cpu_util_slider")
                avg_cpu_util = st.slider("Average CPU Utilization (%)", 1, 100, 45, key="avg_cpu_util_slider")
                cpu_cores_active = st.number_input("Active CPU Cores", min_value=1, max_value=cores, value=min(cores, 12), key="active_cores_input")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="spec-section">', unsafe_allow_html=True)
                st.markdown("**💾 Memory Performance**")
                ram_util = st.slider("Peak RAM Utilization (%)", 1, 100, 80, key="ram_util_slider")
                avg_ram_util = st.slider("Average RAM Utilization (%)", 1, 100, 60, key="avg_ram_util_slider")
                sga_size_gb = st.number_input("SGA/Buffer Pool Size (GB)", min_value=1, max_value=ram, value=min(ram//2, 32), key="sga_size_input")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col3:
                st.markdown('<div class="spec-section">', unsafe_allow_html=True)
                st.markdown("**💿 I/O Performance**")
                max_iops = st.number_input("Peak IOPS", min_value=100, max_value=100000, value=5000, step=100, key="max_iops_input")
                avg_iops = st.number_input("Average IOPS", min_value=100, max_value=max_iops, value=min(max_iops//2, 2000), step=100, key="avg_iops_input")
                max_throughput_mbps = st.number_input("Peak Throughput (MB/s)", min_value=10, max_value=10000, value=250, step=10, key="max_throughput_input")
                st.markdown('</div>', unsafe_allow_html=True)
            
            st.subheader("🌐 Network & Connection Specifications")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown('<div class="spec-section">', unsafe_allow_html=True)
                st.markdown("**🔗 Network Performance**")
                network_bandwidth_gbps = st.number_input("Network Bandwidth (Gbps)", min_value=1, max_value=100, value=10, key="network_bw_input")
                network_latency_ms = st.number_input("Network Latency (ms)", min_value=0.1, max_value=100.0, value=1.0, step=0.1, key="network_latency_input")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="spec-section">', unsafe_allow_html=True)
                st.markdown("**👥 Connection Specifications**")
                max_connections = st.number_input("Max Concurrent Connections", min_value=10, max_value=10000, value=500, step=10, key="max_connections_input")
                avg_connections = st.number_input("Average Concurrent Connections", min_value=10, max_value=max_connections, value=min(max_connections//2, 200), step=10, key="avg_connections_input")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with st.expander("⚙️ Advanced Settings"):
                col1, col2 = st.columns(2)
                
                with col1:
                    enable_encryption = st.checkbox("Enable Encryption at Rest", value=True, key="encryption_checkbox")
                    enable_perf_insights = st.checkbox("Enable Performance Insights", value=True, key="perf_insights_checkbox")
                    enable_enhanced_monitoring = st.checkbox("Enable Enhanced Monitoring", key="enhanced_monitoring_checkbox")
                    
                with col2:
                    monthly_transfer_gb = st.number_input("Monthly Data Transfer (GB)", min_value=0, value=100, key="monthly_transfer_input")
                    backup_retention_override = st.number_input("Backup Retention (days, 0=default)", min_value=0, max_value=35, value=0, key="backup_retention_input")
                    multi_master = st.checkbox("Multi-Master Configuration", key="multi_master_checkbox")
            
            st.subheader("📋 Server Summary")
            st.markdown(f"""
            <div class="server-summary-card">
                <h4>🖥️ {server_name} ({environment})</h4>
                <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin-top: 1rem;">
                    <div><strong>CPU:</strong> {cores} cores @ {cpu_ghz}GHz</div>
                    <div><strong>RAM:</strong> {ram}GB {ram_type}</div>
                    <div><strong>Storage:</strong> {storage}GB {storage_type_hw}</div>
                    <div><strong>IOPS:</strong> {max_iops:,} peak</div>
                </div>
                <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin-top: 0.5rem;">
                    <div><strong>CPU Util:</strong> {cpu_util}% peak</div>
                    <div><strong>RAM Util:</strong> {ram_util}% peak</div>
                    <div><strong>Throughput:</strong> {max_throughput_mbps}MB/s</div>
                    <div><strong>Network:</strong> {network_bandwidth_gbps}Gbps</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("💾 Save Server Specification", type="primary", use_container_width=True):
                server_spec = {
                    'server_name': server_name,
                    'environment': environment,
                    'database_version': database_version,
                    'database_size_gb': database_size_gb,
                    'cores': cores,
                    'cpu_model': cpu_model,
                    'cpu_ghz': cpu_ghz,
                    'ram': ram,
                    'ram_type': ram_type,
                    'ram_speed': ram_speed,
                    'storage': storage,
                    'storage_type': storage_type_hw,
                    'raid_level': raid_level,
                    'growth_rate': growth_rate,
                    'years': years,
                    'cpu_util': cpu_util,
                    'avg_cpu_util': avg_cpu_util,
                    'cpu_cores_active': cpu_cores_active,
                    'ram_util': ram_util,
                    'avg_ram_util': avg_ram_util,
                    'sga_size_gb': sga_size_gb,
                    'max_iops': max_iops,
                    'avg_iops': avg_iops,
                    'max_throughput_mbps': max_throughput_mbps,
                    'network_bandwidth_gbps': network_bandwidth_gbps,
                    'network_latency_ms': network_latency_ms,
                    'max_connections': max_connections,
                    'avg_connections': avg_connections,
                    'enable_encryption': enable_encryption,
                    'enable_perf_insights': enable_perf_insights,
                    'enable_enhanced_monitoring': enable_enhanced_monitoring,
                    'monthly_transfer_gb': monthly_transfer_gb,
                    'backup_retention_override': backup_retention_override,
                    'multi_master': multi_master
                }
                
                st.session_state.current_server_spec = server_spec
                st.success(f"✅ Server specification for {server_name} saved successfully!")
        
        else:
            # Bulk Server Analysis
            st.subheader("📊 Bulk Server Analysis")
            
            st.markdown("""
            <div class="bulk-upload-zone">
                <h3>📁 Upload Server Specifications</h3>
                <p>Upload a CSV or Excel file containing multiple server specifications</p>
            </div>
            """, unsafe_allow_html=True)
            
            uploaded_file = st.file_uploader(
                "Choose CSV or Excel file",
                type=['csv', 'xlsx', 'xls'],
                help="Upload a file with server specifications. Required columns: Server Name, CPU Cores, RAM (GB), Storage (GB)"
            )
            
            with st.expander("📋 Download Template & View Required Format"):
                st.markdown("**Required Columns:**")
                template_data = {
                    'Server_Name': ['PROD-DB-01', 'UAT-DB-01', 'DEV-DB-01'],
                    'CPU_Cores': [16, 8, 4],
                    'RAM_GB': [64, 32, 16],
                    'Storage_GB': [500, 250, 100],
                    'Peak_CPU_Percent': [75, 60, 50],
                    'Peak_RAM_Percent': [80, 70, 60],
                    'Max_IOPS': [5000, 2500, 1000],
                    'Max_Throughput_MBPS': [250, 125, 50],
                    'Database_Engine': ['oracle-ee', 'oracle-ee', 'postgres']
                }
                
                template_df = pd.DataFrame(template_data)
                st.dataframe(template_df, use_container_width=True)
                
                csv_template = template_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV Template",
                    data=csv_template,
                    file_name="server_specifications_template.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            if uploaded_file is not None:
                with st.spinner("Processing uploaded file..."):
                    servers = parse_bulk_upload_file(uploaded_file)
                    
                    if servers:
                        st.session_state.bulk_upload_data = servers
                        
                        st.success(f"✅ Successfully loaded {len(servers)} servers")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        total_cores = sum([s['cpu_cores'] for s in servers])
                        total_ram = sum([s['ram_gb'] for s in servers])
                        total_storage = sum([s['storage_gb'] for s in servers])
                        
                        with col1:
                            st.metric("Total Servers", len(servers))
                        with col2:
                            st.metric("Total CPU Cores", total_cores)
                        with col3:
                            st.metric("Total RAM (GB)", f"{total_ram:,}")
                        with col4:
                            st.metric("Total Storage (GB)", f"{total_storage:,}")
                        
                        st.subheader("📋 Uploaded Servers")
                        servers_df = pd.DataFrame(servers)
                        st.dataframe(servers_df, use_container_width=True)
                        
                        st.session_state.on_prem_servers = servers
            
            with st.expander("➕ Add Individual Server Manually"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    manual_server_name = st.text_input("Server Name", key="manual_server_name")
                    manual_cores = st.number_input("CPU Cores", min_value=1, max_value=128, value=8, key="manual_cores")
                    manual_ram = st.number_input("RAM (GB)", min_value=1, max_value=1024, value=32, key="manual_ram")
                
                with col2:
                    manual_storage = st.number_input("Storage (GB)", min_value=10, value=250, key="manual_storage")
                    manual_cpu_util = st.number_input("Peak CPU (%)", min_value=1, max_value=100, value=70, key="manual_cpu_util")
                    manual_ram_util = st.number_input("Peak RAM (%)", min_value=1, max_value=100, value=75, key="manual_ram_util")
                
                with col3:
                    manual_iops = st.number_input("Max IOPS", min_value=100, value=2500, key="manual_iops")
                    manual_throughput = st.number_input("Max Throughput (MB/s)", min_value=10, value=125, key="manual_throughput")
                    manual_engine = st.selectbox("Database Engine", ["oracle-ee", "oracle-se", "mysql", "postgres"], key="manual_engine")
                
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
            
            if st.session_state.on_prem_servers:
                st.subheader(f"📊 Current Bulk Analysis List ({len(st.session_state.on_prem_servers)} servers)")
                
                bulk_df = pd.DataFrame(st.session_state.on_prem_servers)
                st.dataframe(bulk_df, use_container_width=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("🗑️ Clear Bulk List", use_container_width=True):
                        st.session_state.on_prem_servers = []
                        st.session_state.bulk_upload_data = None
                        st.success("✅ Bulk server list cleared")
                        st.rerun()
                
                with col2:
                    export_csv = bulk_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Export Current List",
                        data=export_csv,
                        file_name=f"bulk_servers_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

# ================================
# TAB 3: SIZING ANALYSIS
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
            # Single Server Analysis
            st.subheader("🖥️ Single Server Analysis")
            
            server_spec = st.session_state.current_server_spec
            
            st.markdown(f"""
            <div class="server-summary-card">
                <h4>🔍 Analyzing: {server_spec['server_name']}</h4>
                <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem;">
                    <div><strong>CPU:</strong> {server_spec['cores']} cores</div>
                    <div><strong>RAM:</strong> {server_spec['ram']}GB</div>
                    <div><strong>Storage:</strong> {server_spec['storage']}GB</div>
                    <div><strong>IOPS:</strong> {server_spec['max_iops']:,}</div>
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
                                "storage_growth_rate": server_spec['growth_rate']/100,
                                "years": server_spec['years'],
                                "enable_encryption": server_spec['enable_encryption'],
                                "enable_perf_insights": server_spec['enable_perf_insights'],
                                "enable_enhanced_monitoring": server_spec['enable_enhanced_monitoring'],
                                "monthly_data_transfer_gb": server_spec['monthly_transfer_gb'],
                                "max_iops": server_spec['max_iops'],
                                "max_throughput_mbps": server_spec['max_throughput_mbps'],
                                "max_connections": server_spec['max_connections']
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
                                        st.warning(f"AI insights generation failed: {e}")
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
            # Bulk Server Analysis
            st.subheader("📊 Bulk Server Analysis")
            
            servers = st.session_state.on_prem_servers
            st.info(f"📋 Ready to analyze {len(servers)} servers")
            
            with st.expander("⚙️ Bulk Analysis Settings"):
                col1, col2 = st.columns(2)
                
                with col1:
                    batch_size = st.number_input("Batch Size", min_value=1, max_value=10, value=3, help="Number of servers to process simultaneously")
                    include_dev_environments = st.checkbox("Include DEV/TEST environments", value=False)
                
                with col2:
                    export_individual_reports = st.checkbox("Export individual server reports", value=False)
                    enable_parallel_processing = st.checkbox("Enable parallel processing", value=True)
            
            if st.button("🚀 Start Bulk Analysis", type="primary", use_container_width=True):
                progress_bar = st.progress(0)
                status_placeholder = st.empty()
                results_placeholder = st.empty()
                
                bulk_results = {}
                total_servers = len(servers)
                
                try:
                    for i, server in enumerate(servers):
                        status_placeholder.text(f"Analyzing {server['server_name']} ({i+1}/{total_servers})")
                        
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
                            
                        except Exception as e:
                            bulk_results[server['server_name']] = {'error': str(e)}
                            st.warning(f"⚠️ Error analyzing {server['server_name']}: {e}")
                        
                        progress_bar.progress((i + 1) / total_servers)
                        
                        if i % 3 == 0:
                            with results_placeholder.container():
                                st.write(f"Completed: {i+1}/{total_servers} servers")
                    
                    st.session_state.bulk_results = bulk_results
                    
                    successful_analyses = len([r for r in bulk_results.values() if 'error' not in r])
                    failed_analyses = total_servers - successful_analyses
                    
                    progress_bar.progress(1.0)
                    status_placeholder.success(f"✅ Bulk analysis complete! {successful_analyses} successful, {failed_analyses} failed")
                    
                    if successful_analyses > 0:
                        st.subheader("📊 Bulk Analysis Results")
                        
                        summary_fig = create_bulk_analysis_summary_chart(bulk_results)
                        if summary_fig:
                            st.plotly_chart(summary_fig, use_container_width=True)
                        
                        summary_data = []
                        total_monthly_cost = 0
                        
                        for server_name, results in bulk_results.items():
                            if 'error' not in results:
                                result = results.get('PROD', list(results.values())[0])
                                if 'error' not in result:
                                    monthly_cost = safe_get(result, 'total_cost', 0)
                                    total_monthly_cost += monthly_cost
                                    
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
                            st.metric("Total Monthly Cost", f"${total_monthly_cost:,.2f}")
                        with col2:
                            st.metric("Total Annual Cost", f"${total_monthly_cost * 12:,.2f}")
                        with col3:
                            avg_cost = total_monthly_cost / successful_analyses if successful_analyses > 0 else 0
                            st.metric("Average Cost per Server", f"${avg_cost:,.2f}")
                    
                except Exception as e:
                    st.error(f"❌ Bulk analysis failed: {str(e)}")
                    st.code(traceback.format_exc())
        
        # Display results for both single and bulk
        if st.session_state.results or st.session_state.bulk_results:
            current_results = st.session_state.results if st.session_state.current_analysis_mode == 'single' else st.session_state.bulk_results
            
            st.subheader("📊 Export Options")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("📊 Export to CSV", use_container_width=True):
                    if st.session_state.current_analysis_mode == 'single':
                        valid_results = {k: v for k, v in current_results.items() if 'error' not in v}
                        
                        df_data = []
                        for env, result in valid_results.items():
                            instance_type_display = safe_get_str(result, 'instance_type', 'N/A')
                            vcpus_display = safe_get(result, 'actual_vCPUs', 0)
                            ram_gb_display = safe_get(result, 'actual_RAM_GB', 0)
                            instance_cost_display = safe_get(result, 'instance_cost', 0)
                            
                            if 'writer' in result:
                                writer_info = result['writer']
                                instance_type_display = safe_get_str(writer_info, 'instance_type', 'N/A')
                                vcpus_display = safe_get(writer_info, 'actual_vCPUs', 0)
                                ram_gb_display = safe_get(writer_info, 'actual_RAM_GB', 0)
                                instance_cost_display = safe_get(writer_info, 'instance_cost', 0)
                                if result['readers']:
                                    for reader_info in result['readers']:
                                        instance_cost_display += safe_get(reader_info, 'instance_cost', 0)
                                    instance_type_display += f" + {len(result['readers'])} Readers ({safe_get_str(result['readers'][0], 'instance_type', 'N/A')})"

                            df_data.append({
                                'Environment': env,
                                'Instance Type': instance_type_display,
                                'vCPUs': vcpus_display,
                                'RAM (GB)': ram_gb_display,
                                'Storage (GB)': safe_get(result, 'storage_GB', 0),
                                'Instance Cost': instance_cost_display,
                                'Storage Cost': safe_get(result, 'storage_cost', 0),
                                'Total Monthly Cost': safe_get(result, 'total_cost', 0)
                            })
                        
                        df = pd.DataFrame(df_data)
                        csv = df.to_csv(index=False)
                        filename = f"rds_single_recommendations_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
                    else:
                        df_data = []
                        for server_name, server_results in current_results.items():
                            if 'error' not in server_results:
                                result = server_results.get('PROD', list(server_results.values())[0])
                                if 'error' not in result:
                                    instance_type_display = ""
                                    vcpus_display = 0
                                    ram_gb_display = 0
                                    instance_cost_display = 0

                                    if 'writer' in result:
                                        writer_info = result['writer']
                                        instance_type_display = safe_get_str(writer_info, 'instance_type', 'N/A')
                                        vcpus_display = safe_get(writer_info, 'actual_vCPUs', 0)
                                        ram_gb_display = safe_get(writer_info, 'actual_RAM_GB', 0)
                                        instance_cost_display = safe_get(writer_info, 'instance_cost', 0)
                                        if result['readers']:
                                            for reader_info in result['readers']:
                                                instance_cost_display += safe_get(reader_info, 'instance_cost', 0)
                                            instance_type_display += f" + {len(result['readers'])} Readers"
                                    else:
                                        instance_type_display = safe_get_str(result, 'instance_type', 'N/A')
                                        vcpus_display = safe_get(result, 'actual_vCPUs', 0)
                                        ram_gb_display = safe_get(result, 'actual_RAM_GB', 0)
                                        instance_cost_display = safe_get(result, 'instance_cost', 0)
                                    
                                    monthly_cost = safe_get(result, 'total_cost', 0)

                                    df_data.append({
                                        'Server Name': server_name,
                                        'Recommended Instance': instance_type_display,
                                        'vCPUs': vcpus_display,
                                        'RAM (GB)': ram_gb_display,
                                        'Storage (GB)': safe_get(result, 'storage_GB', 0),
                                        'Instance Cost': instance_cost_display,
                                        'Storage Cost': safe_get(result, 'storage_cost', 0),
                                        'Total Monthly Cost': monthly_cost,
                                        'Annual Cost': monthly_cost * 12
                                    })
                        
                        df = pd.DataFrame(df_data)
                        csv = df.to_csv(index=False)
                        filename = f"rds_bulk_recommendations_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
                    
                    st.download_button(
                        label="📥 Download CSV",
                        data=csv,
                        file_name=filename,
                        mime="text/csv",
                        use_container_width=True
                    )
            
            with col2:
                if st.button("📋 Export to JSON", use_container_width=True):
                    export_data = {
                        'analysis_mode': st.session_state.current_analysis_mode,
                        'migration_config': {
                            'source_engine': st.session_state.source_engine,
                            'target_engine': st.session_state.target_engine,
                            'migration_type': calculator.migration_profile.migration_type.value if calculator.migration_profile else 'unknown'
                        },
                        'recommendations': current_results,
                        'ai_insights': st.session_state.ai_insights,
                        'generation_time': st.session_state.get('generation_time', 0),
                        'generated_at': datetime.now().isoformat()
                    }
                    
                    if st.session_state.current_analysis_mode == 'single' and 'current_server_spec' in st.session_state:
                        export_data['server_specification'] = st.session_state.current_server_spec
                    elif st.session_state.current_analysis_mode == 'bulk':
                        export_data['bulk_servers'] = st.session_state.on_prem_servers
                    
                    json_str = json.dumps(export_data, indent=2, default=str)
                    
                    report_type = "bulk" if st.session_state.current_analysis_mode == 'bulk' else "single"
                    st.download_button(
                        label="📥 Download JSON",
                        data=json_str,
                        file_name=f"rds_{report_type}_analysis_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                        mime="application/json",
                        use_container_width=True
                    )
            
            with col3:
                if st.button("📈 Export Cost Summary", use_container_width=True):
                    if st.session_state.current_analysis_mode == 'single':
                        valid_results = {k: v for k, v in current_results.items() if 'error' not in v}
                        
                        summary_data = []
                        total_cost = sum([safe_get(r, 'total_cost', 0) for r in valid_results.values()])
                        
                        for env, result in valid_results.items():
                            cost_breakdown = safe_get(result, 'cost_breakdown', {})
                            result_total_cost = safe_get(result, 'total_cost', 0)
                            percentage = (result_total_cost / total_cost * 100) if total_cost > 0 else 0
                            
                            instance_cost_summary = 0
                            if 'writer' in result:
                                instance_cost_summary = safe_get(cost_breakdown, 'writer_monthly', 0) + \
                                                        safe_get(cost_breakdown, 'readers_monthly', 0)
                            else:
                                instance_cost_summary = safe_get(cost_breakdown, 'instance_monthly', 0)

                            summary_data.append({
                                'Environment': env,
                                'Instance Cost': instance_cost_summary,
                                'Storage Cost': safe_get(cost_breakdown, 'storage_monthly', 0),
                                'Backup Cost': safe_get(cost_breakdown, 'backup_monthly', 0),
                                'Features Cost': safe_get(cost_breakdown, 'features_monthly', 0),
                                'Monthly Total': result_total_cost,
                                'Annual Total': result_total_cost * 12,
                                'Percentage of Total': f"{percentage:.1f}%"
                            })
                        
                        filename = f"rds_single_cost_summary_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
                    else:
                        summary_data = []
                        total_monthly_cost = 0
                        
                        for server_name, server_results in current_results.items():
                            if 'error' not in server_results:
                                result = server_results.get('PROD', list(server_results.values())[0])
                                if 'error' not in result:
                                    monthly_cost = safe_get(result, 'total_cost', 0)
                                    total_monthly_cost += monthly_cost
                        
                        for server_name, server_results in current_results.items():
                            if 'error' not in server_results:
                                result = server_results.get('PROD', list(server_results.values())[0])
                                if 'error' not in result:
                                    monthly_cost = safe_get(result, 'total_cost', 0)
                                    percentage = (monthly_cost / total_monthly_cost * 100) if total_monthly_cost > 0 else 0
                                    
                                    instance_type_summary = ""
                                    vcpus_summary = 0
                                    ram_gb_summary = 0
                                    if 'writer' in result:
                                        writer_info = result['writer']
                                        instance_type_summary = safe_get_str(writer_info, 'instance_type', 'N/A')
                                        vcpus_summary = safe_get(writer_info, 'actual_vCPUs', 0)
                                        ram_gb_summary = safe_get(writer_info, 'actual_RAM_GB', 0)
                                        if result['readers']:
                                            instance_type_summary += f" + {len(result['readers'])} Readers"
                                    else:
                                        instance_type_summary = safe_get_str(result, 'instance_type', 'N/A')
                                        vcpus_summary = safe_get(result, 'actual_vCPUs', 0)
                                        ram_gb_summary = safe_get(result, 'actual_RAM_GB', 0)

                                    summary_data.append({
                                        'Server Name': server_name,
                                        'Instance Type': instance_type_summary,
                                        'Monthly Cost': monthly_cost,
                                        'Annual Cost': monthly_cost * 12,
                                        'Percentage of Total': f"{percentage:.1f}%",
                                        'Cost per vCPU': monthly_cost / max(vcpus_summary, 1),
                                        'Cost per GB RAM': monthly_cost / max(ram_gb_summary, 1)
                                    })
                        
                        filename = f"rds_bulk_cost_summary_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
                    
                    summary_df = pd.DataFrame(summary_data)
                    csv_summary = summary_df.to_csv(index=False)
                    
                    st.download_button(
                        label="📥 Download Cost Summary",
                        data=csv_summary,
                        file_name=filename,
                        mime="text/csv",
                        use_container_width=True
                    )
            
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

    # Data Transfer Analysis Section
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
                        reasoning = f"Saves {time_savings/24:.1f} days for only ${cost_difference:.0f} additional cost"
                    elif cost_difference < 0:  # DX is actually cheaper
                        recommendation = "✅ **Recommended: DataSync over Direct Connect**"
                        reasoning = "Faster AND cheaper than internet transfer"
                    else:
                        recommendation = "🌐 **Consider: DataSync over Internet**"
                        reasoning = f"More cost-effective option (saves ${cost_difference:.0f})"
                    
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
    if st.session_state.current_analysis_mode == 'bulk':
        if st.session_state.on_prem_servers:
            # Simple bulk transfer analysis
            st.info("🔄 Bulk transfer analysis: Calculate individual server transfers and sum the results")
            
            servers = st.session_state.on_prem_servers
            total_data_gb = sum([server['storage_gb'] for server in servers]) * 1.2  # Include logs
            
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
                        
                        for server in servers:
                            server_data_gb = server['storage_gb'] * 1.2
                            
                            # Calculate effective bandwidth (divided if parallel)
                            effective_dx_gbps = bulk_dx_gbps / max_concurrent if parallel_transfers else bulk_dx_gbps
                            effective_internet_mbps = 1000 / max_concurrent if parallel_transfers else 1000
                            
                            server_results = transfer_calculator.calculate_comprehensive_transfer_analysis(
                                data_size_gb=server_data_gb,
                                region=st.session_state.region,
                                dx_bandwidth_gbps=effective_dx_gbps,
                                internet_bandwidth_mbps=effective_internet_mbps,
                                compression_type='database'
                            )
                            
                            # Accumulate costs and times
                            dx_result = server_results['datasync_dx']
                            internet_result = server_results['datasync_internet']
                            
                            total_cost_dx += dx_result.total_cost
                            total_cost_internet += internet_result.total_cost
                            
                            if parallel_transfers:
                                max_time_dx = max(max_time_dx, dx_result.transfer_time_days)
                                max_time_internet = max(max_time_internet, internet_result.transfer_time_days)
                            else:
                                max_time_dx += dx_result.transfer_time_days
                                max_time_internet += internet_result.transfer_time_days
                        
                        # Display bulk results
                        st.subheader("📊 Bulk Transfer Results")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Total Data", f"{total_data_gb:,.0f} GB")
                        
                        with col2:
                            st.metric("DX Transfer Time", f"{max_time_dx:.1f} days")
                        
                        with col3:
                            st.metric("DX Total Cost", f"${total_cost_dx:,.0f}")
                        
                        with col4:
                            savings = max_time_internet - max_time_dx
                            st.metric("Time Saved", f"{savings:.1f} days")
                        
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
                        
                        # Store bulk results
                        st.session_state.bulk_transfer_summary = {
                            'total_data_gb': total_data_gb,
                            'dx_time_days': max_time_dx,
                            'dx_cost': total_cost_dx,
                            'internet_time_days': max_time_internet,
                            'internet_cost': total_cost_internet,
                            'parallel_enabled': parallel_transfers,
                            'server_count': len(servers)
                        }
                        
                    except Exception as e:
                        st.error(f"❌ Error in bulk transfer calculation: {str(e)}")
        else:
            st.info("💡 Upload or add server specifications for bulk analysis first.")

    # If no transfer calculator available
    if not transfer_calculator:
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
            total_monthly_cost = 0
            total_annual_cost = 0
            server_costs = []
            
            for server_name, server_results in current_results.items():
                if 'error' not in server_results:
                    successful_servers += 1
                    result = server_results.get('PROD', list(server_results.values())[0])
                    if 'error' not in result:
                        monthly_cost = safe_get(result, 'total_cost', 0)
                        total_monthly_cost += monthly_cost
                        total_annual_cost += monthly_cost * 12
                        
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
                        <div class="metric-value">${total_monthly_cost:,.0f}</div>
                        <div class="metric-label">Total Monthly Cost (All Servers)</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col2:
                    st.markdown(f"""
                    <div class="metric-container">
                        <div class="metric-value">${total_annual_cost:,.0f}</div>
                        <div class="metric-label">Total Annual Cost (All Servers)</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col3:
                    avg_cost_per_server = total_monthly_cost / successful_servers
                    st.markdown(f"""
                    <div class="metric-container">
                        <div class="metric-value">${avg_cost_per_server:,.0f}</div>
                        <div class="metric-label">Avg. Monthly Cost per Server</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.subheader("📈 Bulk Analysis Visualizations")
                bulk_summary_fig = create_bulk_analysis_summary_chart(current_results)
                if bulk_summary_fig:
                    st.plotly_chart(bulk_summary_fig, use_container_width=True)
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
        if st.session_state.results:
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
    
    if not st.session_state.ai_insights:
        st.info("💡 Generate sizing recommendations first to enable AI insights.")
    else:
        ai_insights = st.session_state.ai_insights
        
        if "error" in ai_insights:
            st.error(f"❌ Error retrieving AI insights: {ai_insights['error']}")
        else:
            st.markdown("""
            <div class="ai-insight-card">
                <h3>🤖 AI-Powered Analysis from Claude</h3>
                <p>Leveraging advanced AI to provide deeper insights into your migration.</p>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                <div class="metric-container">
                    <div class="metric-value">{ai_insights.get('risk_level', 'N/A')}</div>
                    <div class="metric-label">Migration Risk Level</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                cost_opt_potential = ai_insights.get('cost_optimization_potential', 0) * 100
                st.markdown(f"""
                <div class="metric-container">
                    <div class="metric-value">{cost_opt_potential:.0f}%</div>
                    <div class="metric-label">Cost Optimization Potential</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                writers = ai_insights.get('recommended_writers', 'N/A')
                readers = ai_insights.get('recommended_readers', 'N/A')
                if writers != 'N/A' and readers != 'N/A':
                    st.markdown(f"""
                    <div class="metric-container">
                        <div class="metric-value">{writers}W / {readers}R</div>
                        <div class="metric-label">AI Recommended Arch.</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                     st.markdown(f"""
                    <div class="metric-container">
                        <div class="metric-value">N/A</div>
                        <div class="metric-label">AI Recommended Arch.</div>
                    </div>
                    """, unsafe_allow_html=True)

            st.subheader("Comprehensive AI Analysis")
            st.markdown('<div class="advisory-box">', unsafe_allow_html=True)
            st.write(ai_insights.get("ai_analysis", "No detailed AI analysis available."))
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.subheader("Recommended Migration Phases")
            if ai_insights.get("recommended_migration_phases"):
                st.markdown('<div class="phase-timeline">', unsafe_allow_html=True)
                for i, phase in enumerate(ai_insights["recommended_migration_phases"]):
                    st.markdown(f"**Phase {i+1}:** {phase}")
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.info("No specific migration phases recommended by AI.")

# ================================
# TAB 6: REPORTS
# ================================
# In TAB 6: Reports section, around line 1800-2000
# Replace the existing bulk report generation code with:

with tab6:
    st.header("📋 Export & Reporting")
    
    if st.session_state.current_analysis_mode == 'bulk' and st.session_state.on_prem_servers:
        
        # Add chunked processing option for large bulk reports
        if len(st.session_state.on_prem_servers) > 20:
            st.info(f"📊 Large dataset detected ({len(st.session_state.on_prem_servers)} servers). Using chunked processing for optimal performance.")
            
            chunk_size = st.slider(
                "Chunk Size (servers per batch)", 
                min_value=5, 
                max_value=25, 
                value=10,
                help="Smaller chunks use less memory but take longer"
            )
            
            if st.button("📄 Generate Bulk Report (Chunked)", type="primary", use_container_width=True):
                with st.spinner("Generating bulk report in chunks..."):
                    try:
                        # Use chunked processing
                        enhanced_generator = st.session_state.enhanced_report_generator
                        processed_chunks = enhanced_generator.generate_bulk_report_in_chunks(
                            st.session_state.on_prem_servers, 
                            chunk_size=chunk_size
                        )
                        
                        # Combine results from all chunks
                        all_results = {}
                        for chunk in processed_chunks:
                            for server_result in chunk:
                                if 'error' not in server_result:
                                    server_name = server_result.get('server_name', f'Server_{len(all_results)}')
                                    all_results[server_name] = server_result
                        
                        # Generate final report
                        st.session_state.bulk_results = all_results
                        st.success(f"✅ Bulk report generated successfully! Processed {len(all_results)} servers.")
                        
                    except Exception as e:
                        st.error(f"❌ Error generating chunked bulk report: {str(e)}")
        
        else:
            # Use regular processing for smaller datasets
            if st.button("📄 Generate Standard Bulk Report", type="primary", use_container_width=True):
                # Your existing bulk processing code here
                pass
        # Additional export options for reports tab
        st.subheader("📊 Additional Export Options")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**📈 Executive Summary**")
            if st.button("Generate Executive Summary", use_container_width=True):
                if st.session_state.current_analysis_mode == 'single':
                    if st.session_state.results:
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
                else:
                    st.info("Executive summary available for single server analysis only.")
        
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
                else:
                    st.info("Technical specifications available for single server analysis only.")
        
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

    # Transfer Analysis Report
    if hasattr(st.session_state, 'transfer_results') and st.session_state.transfer_results:
        st.subheader("🚛 Transfer Analysis Report")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📊 Export Transfer Analysis", use_container_width=True):
                transfer_export_data = {
                    'transfer_analysis': {
                        'data_size_gb': st.session_state.transfer_data_size,
                        'analysis_date': datetime.now().isoformat(),
                        'configuration': {
                            'dx_bandwidth': st.session_state.get('dx_bandwidth', '10Gbps'),
                            'internet_bandwidth_mbps': st.session_state.get('internet_bandwidth', 100),
                            'compression_type': st.session_state.get('compression_type', 'database'),
                            'time_sensitivity': st.session_state.get('time_sensitivity', 'medium'),
                            'cost_sensitivity': st.session_state.get('cost_sensitivity', 'medium')
                        },
                        'results': {}
                    }
                }
                
                for method, result in st.session_state.transfer_results.items():
                    transfer_export_data['transfer_analysis']['results'][method] = {
                        'recommended_method': result.recommended_method,
                        'transfer_time_hours': result.transfer_time_hours,
                        'transfer_time_days': result.transfer_time_days,
                        'total_cost': result.total_cost,
                        'cost_breakdown': result.cost_breakdown,
                        'bandwidth_utilization': result.bandwidth_utilization,
                        'estimated_downtime_hours': result.estimated_downtime_hours
                    }
                
                transfer_json = json.dumps(transfer_export_data, indent=2, default=str)
                
                st.download_button(
                    label="📥 Download Transfer Analysis",
                    data=transfer_json,
                    file_name=f"transfer_analysis_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                    mime="application/json",
                    use_container_width=True
                )
        
        with col2:
            if st.button("📋 Export Transfer Summary CSV", use_container_width=True):
                transfer_summary_data = []
                
                for method, result in st.session_state.transfer_results.items():
                    transfer_summary_data.append({
                        'Transfer Method': result.recommended_method,
                        'Data Size (GB)': st.session_state.transfer_data_size,
                        'Transfer Time (Hours)': result.transfer_time_hours,
                        'Transfer Time (Days)': result.transfer_time_days,
                        'Total Cost': result.total_cost,
                        'Bandwidth Utilization (%)': result.bandwidth_utilization,
                        'Estimated Downtime (Hours)': result.estimated_downtime_hours,
                        'Primary Cost Component': max(result.cost_breakdown.keys(), key=lambda k: result.cost_breakdown[k]) if result.cost_breakdown else 'N/A'
                    })
                
                transfer_summary_df = pd.DataFrame(transfer_summary_data)
                transfer_csv = transfer_summary_df.to_csv(index=False)
                
                st.download_button(
                    label="📥 Download Transfer Summary",
                    data=transfer_csv,
                    file_name=f"transfer_summary_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )

# ================================
# FOOTER
# ================================

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <h5>🚀 Enterprise AWS RDS Migration & Sizing Tool v2.0</h5>
    <p>AI-Powered Database Migration Analysis • Built for Enterprise Scale</p>
    <p>💡 For support and advanced features, contact your AWS solutions architect</p>
</div>
""", unsafe_allow_html=True)