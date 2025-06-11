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
            # Show available test users (remove this in production)
            if st.form_submit_button("👥 Show Test Users", use_container_width=True):
                st.session_state.show_test_users = True
    
    if login_button:
        if email and password:
            with st.spinner("Authenticating..."):
                user_data = authenticate_user(email, password)
                
                if user_data['authenticated']:
                    # Create session token
                    token = create_session_token(user_data)
                    if token:
                        # Store in session state
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
    
    # Show test users for development (remove in production)
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
    # Clear all authentication-related session state
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
    # Initialize authentication state
    if 'user_authenticated' not in st.session_state:
        st.session_state.user_authenticated = False
    
    # Check for existing session token
    if not st.session_state.user_authenticated and 'session_token' in st.session_state:
        token_data = verify_session_token(st.session_state.session_token)
        if token_data['authenticated']:
            st.session_state.user_authenticated = True
            st.session_state.user_data = token_data
            st.session_state.user_id = token_data['username']
            st.session_state.user_email = token_data['email']
            st.session_state.is_logged_in = True
        else:
            # Clear invalid session
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

def check_user_role(required_role: str) -> bool:
    """Check if current user has required role"""
    if not st.session_state.get('user_authenticated', False):
        return False
    
    user_role = st.session_state.user_data.get('role', '')
    
    # Admin has access to everything
    if user_role == 'admin':
        return True
    
    # Check specific role
    return user_role == required_role

def generate_test_passwords():
    """Generate hashed passwords for your users - Run this once to get hashes"""
    test_passwords = {
        'admin': 'AdminPass123!',
        'analyst': 'AnalystPass456!', 
        'manager': 'ManagerPass789!'
    }
    
    st.subheader("🔑 Password Hash Generator")
    st.markdown("Use these hashed passwords in your secrets.toml file:")
    
    for username, plain_password in test_passwords.items():
        hashed = hash_password(plain_password)
        st.code(f"{username} password: {plain_password}")
        st.code(f"Hashed: {hashed}")
        st.markdown("---")

# ================================
# AUTHENTICATION CHECK
# ================================

# Check authentication before showing the app
if not check_authentication():
    show_login_form()
    st.stop()  # Don't show the rest of the app until authenticated

# Enhanced Custom CSS for better UI
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
    
    .cost-breakdown {
        background: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
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
        text_align: center;
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

# Helper function to safely get values from results dictionary
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
    st.session_state.deployment_option = "Multi-AZ" # Default to Multi-AZ
if 'bulk_results' not in st.session_state:
    st.session_state.bulk_results = {}
if 'on_prem_servers' not in st.session_state:
    st.session_state.on_prem_servers = []
if 'bulk_upload_data' not in st.session_state:
    st.session_state.bulk_upload_data = None
if 'current_analysis_mode' not in st.session_state:
    st.session_state.current_analysis_mode = 'single'  # 'single' or 'bulk'

# Firebase session state variables
if 'firebase_app' not in st.session_state:
    st.session_state.firebase_app = None
if 'firebase_auth' not in st.session_state:
    st.session_state.firebase_auth = None
if 'firebase_db' not in st.session_state:
    st.session_state.firebase_db = None

# CORRECTED Firebase initialization function
@st.cache_resource(ttl=3600)
def initialize_firebase():
    """Initializes the Firebase app and authenticates the user."""
    try:
        # Check if Firebase is already initialized
        if firebase_admin._apps:
            st.info("Firebase already initialized")
            app = firebase_admin.get_app()
            return app, auth, firestore.client()
        
        # Load Firebase config from Streamlit secrets
        if "connections" not in st.secrets or "firebase" not in st.secrets["connections"]:
            st.error("Firebase configuration not found in Streamlit secrets.")
            return None, None, None

        # Convert config to dictionary
        firebase_config_dict = dict(st.secrets["connections"]["firebase"])
        
        # Validate required fields
        required_fields = ['project_id', 'private_key', 'client_email']
        missing_fields = [field for field in required_fields if not firebase_config_dict.get(field)]
        
        if missing_fields:
            st.error(f"Missing required Firebase fields: {missing_fields}")
            return None, None, None

        # Set service account type
        firebase_config_dict['type'] = 'service_account'
        
        # Validate private key format
        private_key = firebase_config_dict.get('private_key', '').strip()
        
        if not private_key.startswith('-----BEGIN PRIVATE KEY-----'):
            st.error("Private key missing BEGIN header")
            return None, None, None
        
        if not private_key.endswith('-----END PRIVATE KEY-----'):
            st.error("Private key missing END footer")
            return None, None, None

        # Initialize Firebase
        cred = credentials.Certificate(firebase_config_dict)
        firebase_app = firebase_admin.initialize_app(
            cred, 
            options={'projectId': firebase_config_dict['project_id']}
        )
        
        # Get Firestore client
        db_client = firestore.client(firebase_app)
        
        st.success("🎉 Firebase Admin SDK initialized successfully!")
        
        # Return: app, auth module, firestore client
        return firebase_app, auth, db_client
        
    except Exception as e:
        st.error(f"Firebase initialization failed: {e}")
        import traceback
        st.code(traceback.format_exc())
        return None, None, None

# Debug function for Firebase configuration
def debug_firebase_config():
    """Debug Firebase configuration to identify issues"""
    st.subheader("🔧 Firebase Configuration Debug")
    
    try:
        if "connections" not in st.secrets:
            st.error("❌ No 'connections' section in secrets")
            return
            
        if "firebase" not in st.secrets["connections"]:
            st.error("❌ No 'firebase' section in connections")
            return
            
        config = dict(st.secrets["connections"]["firebase"])
        st.success("✅ Firebase config section found")
        
        # Check individual fields
        required_fields = ['project_id', 'private_key', 'client_email', 'private_key_id']
        
        st.write("**Field validation:**")
        
        for field in required_fields:
            if field in config and config[field]:
                if field == 'private_key':
                    key = config[field]
                    st.write(f"✅ {field}: Present ({len(key)} characters)")
                    st.write(f"   - Starts correctly: {key.startswith('-----BEGIN PRIVATE KEY-----')}")
                    st.write(f"   - Ends correctly: {key.endswith('-----END PRIVATE KEY-----')}")
                else:
                    value = str(config[field])[:50] + "..." if len(str(config[field])) > 50 else str(config[field])
                    st.write(f"✅ {field}: `{value}`")
            else:
                st.error(f"❌ {field}: Missing or empty")
        
        # Test credential creation
        st.write("**Credential creation test:**")
        try:
            test_config = dict(config)
            test_config['type'] = 'service_account'
            
            # Don't actually create credentials in test, just validate structure
            if all(test_config.get(field) for field in required_fields):
                st.success("✅ All required fields present for credential creation")
            else:
                st.error("❌ Missing required fields for credential creation")
                
        except Exception as test_error:
            st.error(f"❌ Credential validation failed: {test_error}")
    
    except Exception as e:
        st.error(f"❌ Debug failed: {e}")

# Initialize Firebase and store in session state
if st.session_state.firebase_app is None:
    st.session_state.firebase_app, st.session_state.firebase_auth, st.session_state.firebase_db = initialize_firebase()

# Enhanced visualization functions
def create_cost_heatmap(results):
    """Create cost heatmap for environment comparison"""
    if not results:
        return None
    
    valid_results = {k: v for k, v in results.items() if 'error' not in v}
    if not valid_results:
        return None
    
    environments = list(valid_results.keys())
    # Adjust cost categories based on new structure (if Multi-AZ)
    # This assumes 'total_cost' is always available at the top level
    cost_categories = ['Instance Cost', 'Storage Cost', 'Backup Cost', 'Total Cost']
    
    cost_matrix = []
    for env in environments:
        result = valid_results[env]
        # Check if it's a Multi-AZ breakdown or single instance
        if 'writer' in result and 'readers' in result:
            instance_cost_sum = safe_get(result['cost_breakdown'], 'writer_monthly', 0) + \
                                safe_get(result['cost_breakdown'], 'readers_monthly', 0)
            storage_cost = safe_get(result['cost_breakdown'], 'storage_monthly', 0)
            backup_cost = safe_get(result['cost_breakdown'], 'backup_monthly', 0)
            total_cost = safe_get(result, 'total_cost', 0)
        else:
            # Old/single instance structure
            cost_breakdown = safe_get(result, 'cost_breakdown', {})
            instance_cost_sum = safe_get(cost_breakdown, 'instance_monthly', safe_get(result, 'instance_cost', 0))
            storage_cost = safe_get(cost_breakdown, 'storage_monthly', safe_get(result, 'storage_cost', 0))
            backup_cost = safe_get(cost_breakdown, 'backup_monthly', safe_get(result, 'storage_cost', 0) * 0.25)
            total_cost = safe_get(result, 'total_cost', 0)

        row = [
            instance_cost_sum,
            storage_cost,
            backup_cost,
            total_cost
        ]
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

def create_migration_complexity_chart(calculator):
    """Create migration complexity visualization"""
    if not calculator or not calculator.migration_profile:
        return None
    
    profile = calculator.migration_profile
    
    # Create complexity factors visualization
    factors = {
        'Schema Complexity': profile.complexity_factor,
        'Feature Compatibility': profile.feature_compatibility,
        'Data Type Compatibility': 0.9,  # Placeholder
        'Performance Impact': 1.2,  # Placeholder
    }
    
    fig = go.Figure(go.Bar(
        x=list(factors.keys()),
        y=list(factors.values()),
        marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    ))
    
    fig.update_layout(
        title='🔄 Migration Complexity Factors',
        yaxis_title='Factor Score',
        height=350
    )
    
    return fig

def create_workload_distribution_pie(workload_chars):
    """Create workload characteristics pie chart"""
    if not workload_chars:
        return None
    
    # Map workload characteristics to percentages
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
    instance_types = [] # This will be a list of instance types (writer or main)
    vcpus = [] # This will be for the main instance (writer or single)
    ram_gb = [] # This will be for the main instance (writer or single)
    
    for server_name, results in bulk_results.items():
        if 'error' not in results:
            # Get PROD environment or first available
            result = results.get('PROD', list(results.values())[0])
            if 'error' not in result:
                server_names.append(server_name)
                total_costs.append(safe_get(result, 'total_cost', 0))
                
                # Adapt for new structure: get writer type or single instance type
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
    
    # Create subplot with secondary y-axis
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Monthly Cost by Server', 'vCPUs by Server', 'RAM by Server', 'Instance Type Distribution'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"type": "pie"}]]
    )
    
    # Monthly costs
    fig.add_trace(
        go.Bar(x=server_names, y=total_costs, name='Monthly Cost', marker_color='#1f77b4'),
        row=1, col=1
    )
    
    # vCPUs
    fig.add_trace(
        go.Bar(x=server_names, y=vcpus, name='vCPUs', marker_color='#ff7f0e'),
        row=1, col=2
    )
    
    # RAM
    fig.add_trace(
        go.Bar(x=server_names, y=ram_gb, name='RAM (GB)', marker_color='#2ca02c'),
        row=2, col=1
    )
    
    # Instance type distribution
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
        
        # Standardize column names (case insensitive)
        df.columns = df.columns.str.lower().str.strip()
        
        # Expected columns mapping
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
        
        # Map columns
        mapped_columns = {}
        for standard_name, possible_names in column_mapping.items():
            for col in df.columns:
                if col in possible_names:
                    mapped_columns[standard_name] = col
                    break
        
        # Validate required columns
        required_columns = ['server_name', 'cpu_cores', 'ram_gb', 'storage_gb']
        missing_columns = [col for col in required_columns if col not in mapped_columns]
        
        if missing_columns:
            st.error(f"Missing required columns: {missing_columns}")
            st.info("Required columns: Server Name, CPU Cores, RAM (GB), Storage (GB)")
            return None
        
        # Extract and clean data
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
                
                # Validate server data
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

# Add debug button in sidebar
if st.sidebar.button("🔧 Debug Firebase"):
    debug_firebase_config()

# Add password generator for development (remove in production)
if st.sidebar.button("🔧 Generate Password Hashes (Dev Only)"):
    with st.sidebar.expander("Password Generator"):
        generate_test_passwords()

st.markdown("---") # Visual separator after auth status

# Main navigation
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["🎯 Migration Planning", "🖥️ Server Specifications", "📊 Sizing Analysis", "💰 Financial Analysis", "🤖 AI Insights", "📋 Reports"])

with tab1:
    st.header("Migration Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔄 Migration Type")
        
        # Migration path selection
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
        
        # Determine migration type
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
    
    # AWS Configuration
    st.subheader("☁️ AWS Configuration")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        region = st.selectbox("AWS Region", ["us-east-1", "us-west-1", "us-west-2", "eu-west-1", "eu-central-1", "ap-southeast-1"], key="region_select")
    
    with col2:
        # Initializing the deployment option with session state for persistence
        st.session_state.deployment_option = st.selectbox(
            "Deployment Option", 
            ["Single-AZ", "Multi-AZ", "Multi-AZ Cluster", "Aurora Global", "Serverless"], 
            index=["Single-AZ", "Multi-AZ", "Multi-AZ Cluster", "Aurora Global", "Serverless"].index(st.session_state.deployment_option),
            key="deployment_option_select"
        )
    
    with col3:
        storage_type = st.selectbox("Storage Type", ["gp3", "gp2", "io1", "io2", "aurora"], key="storage_type_select")
    
    # Store selected values
    st.session_state.source_engine = source_engine_selection
    st.session_state.target_engine = target_engine_selection
    # st.session_state.deployment_option is already updated by the selectbox
    st.session_state.region = region
    st.session_state.storage_type = storage_type

    if st.button("🎯 Configure Migration", type="primary", use_container_width=True):
        with st.spinner("Configuring migration parameters..."):
            try:
                # Update API key if needed
                if st.session_state.user_claude_api_key_input and st.session_state.calculator.ai_client is None:
                    st.session_state.calculator = EnhancedRDSSizingCalculator(
                        anthropic_api_key=st.session_state.user_claude_api_key_input,
                        use_real_time_pricing=True
                    )
                
                # Configure workload characteristics
                workload_chars = WorkloadCharacteristics(
                    cpu_utilization_pattern=cpu_pattern,
                    memory_usage_pattern=memory_pattern,
                    io_pattern=io_pattern,
                    connection_count=connection_count,
                    transaction_volume=transaction_volume,
                    analytical_workload=analytical_workload
                )
                
                # Set migration parameters
                st.session_state.calculator.set_migration_parameters(
                    source_engine_selection, target_engine_selection, workload_chars
                )
                
                st.session_state.migration_configured = True
                
                # Display migration analysis
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

with tab2:
    st.header("🖥️ On-Premises Server Specifications")
    
    if not st.session_state.migration_configured:
        st.warning("⚠️ Please configure migration settings in the Migration Planning tab first.")
    else:
        # Analysis Mode Selection
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
            
            # Server Basic Info
            with st.expander("📋 Server Information", expanded=True):
                col1, col2 = st.columns(2)
                with col1:
                    server_name = st.text_input("Server Name/Hostname", value="PROD-DB-01", key="server_name_input")
                    environment = st.selectbox("Environment", ["PROD", "UAT", "DEV", "TEST"], key="environment_select")
                with col2:
                    database_version = st.text_input("Database Version", value="12.1.0.2", key="db_version_input")
                    database_size_gb = st.number_input("Database Size (GB)", min_value=1, value=500, key="db_size_input")
            
            # Comprehensive Hardware Specifications
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
                storage_type = st.selectbox("Storage Type", ["HDD", "SSD", "NVMe SSD", "SAN"], index=2, key="storage_type_hw_select")
                raid_level = st.selectbox("RAID Level", ["RAID 0", "RAID 1", "RAID 5", "RAID 10", "RAID 6"], index=3, key="raid_level_select")
                st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown('<div class="spec-section">', unsafe_allow_html=True)
                st.markdown("**📈 Growth & Planning**")
                growth_rate = st.number_input("Annual Growth Rate (%)", min_value=0, max_value=100, value=20, key="growth_rate_input")
                years = st.number_input("Planning Horizon (years)", min_value=1, max_value=5, value=3, key="years_input")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Performance Specifications
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
            
            # Network and Connection Specifications
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
            
            # Advanced settings
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
            
            # Server Summary Card
            st.subheader("📋 Server Summary")
            st.markdown(f"""
            <div class="server-summary-card">
                <h4>🖥️ {server_name} ({environment})</h4>
                <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin-top: 1rem;">
                    <div><strong>CPU:</strong> {cores} cores @ {cpu_ghz}GHz</div>
                    <div><strong>RAM:</strong> {ram}GB {ram_type}</div>
                    <div><strong>Storage:</strong> {storage}GB {storage_type}</div>
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
            
            # Save server specification
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
                    'storage_type': storage_type,
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
                
                # Store in session state for analysis
                st.session_state.current_server_spec = server_spec
                st.success(f"✅ Server specification for {server_name} saved successfully!")
        
        else:
            # Bulk Server Analysis
            st.subheader("📊 Bulk Server Analysis")
            
            # File upload section
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
            
            # Display template
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
                
                # Download template
                csv_template = template_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV Template",
                    data=csv_template,
                    file_name="server_specifications_template.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            # Process uploaded file
            if uploaded_file is not None:
                with st.spinner("Processing uploaded file..."):
                    servers = parse_bulk_upload_file(uploaded_file)
                    
                    if servers:
                        st.session_state.bulk_upload_data = servers
                        
                        # Display upload summary
                        st.success(f"✅ Successfully loaded {len(servers)} servers")
                        
                        # Show summary statistics
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
                        
                        # Display server list
                        st.subheader("📋 Uploaded Servers")
                        servers_df = pd.DataFrame(servers)
                        st.dataframe(servers_df, use_container_width=True)
                        
                        # Store for analysis
                        st.session_state.on_prem_servers = servers
            
            # Manual server addition (for bulk mode)
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
            
            # Display current bulk server list
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
                    # Export current list
                    export_csv = bulk_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Export Current List",
                        data=export_csv,
                        file_name=f"bulk_servers_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

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
            
            # Display current server info
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
            
            # Generate recommendations button
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
                                "deployment": st.session_state.deployment_option, # Pass deployment option
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
                            
                            # Generate recommendations
                            results = calculator.generate_comprehensive_recommendations(inputs)
                            st.session_state.results = results
                            st.session_state.generation_time = time.time() - start_time
                            
                            # Generate AI insights if available
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
            
            # Bulk analysis settings
            with st.expander("⚙️ Bulk Analysis Settings"):
                col1, col2 = st.columns(2)
                
                with col1:
                    batch_size = st.number_input("Batch Size", min_value=1, max_value=10, value=3, help="Number of servers to process simultaneously")
                    include_dev_environments = st.checkbox("Include DEV/TEST environments", value=False)
                
                with col2:
                    export_individual_reports = st.checkbox("Export individual server reports", value=False)
                    enable_parallel_processing = st.checkbox("Enable parallel processing", value=True)
            
            # Start bulk analysis
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
                                "deployment": st.session_state.deployment_option, # Pass deployment option
                                "storage_type": st.session_state.storage_type,
                                "on_prem_cores": server['cpu_cores'],
                                "peak_cpu_percent": server['peak_cpu_percent'],
                                "on_prem_ram_gb": server['ram_gb'],
                                "peak_ram_percent": server['peak_ram_percent'],
                                "storage_current_gb": server['storage_gb'],
                                "storage_growth_rate": 0.2,  # Default 20%
                                "years": 3,  # Default 3 years
                                "enable_encryption": True,
                                "enable_perf_insights": True,
                                "enable_enhanced_monitoring": False,
                                "monthly_data_transfer_gb": 100,
                                "max_iops": server['max_iops'],
                                "max_throughput_mbps": server['max_throughput_mbps']
                            }
                            
                            # Generate recommendations
                            server_results = calculator.generate_comprehensive_recommendations(inputs)
                            bulk_results[server['server_name']] = server_results
                            
                        except Exception as e:
                            bulk_results[server['server_name']] = {'error': str(e)}
                            st.warning(f"⚠️ Error analyzing {server['server_name']}: {e}")
                        
                        # Update progress
                        progress_bar.progress((i + 1) / total_servers)
                        
                        # Brief display of intermediate results
                        if i % 3 == 0:  # Update display every 3 servers
                            with results_placeholder.container():
                                st.write(f"Completed: {i+1}/{total_servers} servers")
                    
                    # Store results
                    st.session_state.bulk_results = bulk_results
                    
                    # Final success message
                    successful_analyses = len([r for r in bulk_results.values() if 'error' not in r])
                    failed_analyses = total_servers - successful_analyses
                    
                    progress_bar.progress(1.0)
                    status_placeholder.success(f"✅ Bulk analysis complete! {successful_analyses} successful, {failed_analyses} failed")
                    
                    # Display bulk results summary
                    if successful_analyses > 0:
                        st.subheader("📊 Bulk Analysis Results")
                        
                        # Create summary chart
                        summary_fig = create_bulk_analysis_summary_chart(bulk_results)
                        if summary_fig:
                            st.plotly_chart(summary_fig, use_container_width=True)
                        
                        # Create summary table
                        summary_data = []
                        total_monthly_cost = 0
                        
                        for server_name, results in bulk_results.items():
                            if 'error' not in results:
                                # Get PROD environment or first available
                                result = results.get('PROD', list(results.values())[0])
                                if 'error' not in result:
                                    monthly_cost = safe_get(result, 'total_cost', 0)
                                    total_monthly_cost += monthly_cost
                                    
                                    # Adapt for new structure: get writer type or single instance type
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
                        
                        # Cost summary
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
        
        # Display results (both single and bulk)
        if st.session_state.results or st.session_state.bulk_results:
            current_results = st.session_state.results if st.session_state.current_analysis_mode == 'single' else st.session_state.bulk_results
            ai_insights = st.session_state.ai_insights
            
            # Export options
            st.subheader("📊 Export Options")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("📊 Export to CSV", use_container_width=True):
                    if st.session_state.current_analysis_mode == 'single':
                        # Single server CSV export
                        valid_results = {k: v for k, v in current_results.items() if 'error' not in v}
                        
                        df_data = []
                        for env, result in valid_results.items():
                            # Adapt for new structure
                            instance_type_display = safe_get_str(result, 'instance_type', 'N/A')
                            vcpus_display = safe_get(result, 'actual_vCPUs', 0)
                            ram_gb_display = safe_get(result, 'actual_RAM_GB', 0)
                            instance_cost_display = safe_get(result, 'instance_cost', 0)
                            
                            if 'writer' in result: # Multi-AZ structure
                                writer_info = result['writer']
                                instance_type_display = safe_get_str(writer_info, 'instance_type', 'N/A')
                                vcpus_display = safe_get(writer_info, 'actual_vCPUs', 0)
                                ram_gb_display = safe_get(writer_info, 'actual_RAM_GB', 0)
                                instance_cost_display = safe_get(writer_info, 'instance_cost', 0) # Writer instance cost
                                if result['readers']:
                                    for reader_info in result['readers']:
                                        instance_cost_display += safe_get(reader_info, 'instance_cost', 0) # Add reader costs
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
                        # Bulk server CSV export
                        df_data = []
                        for server_name, server_results in current_results.items():
                            if 'error' not in server_results:
                                result = server_results.get('PROD', list(server_results.values())[0])
                                if 'error' not in result:
                                    # Adapt for new structure
                                    instance_type_display = ""
                                    vcpus_display = 0
                                    ram_gb_display = 0
                                    instance_cost_display = 0

                                    if 'writer' in result: # Multi-AZ structure
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
                        'ai_insights': ai_insights,
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
                        # Single server cost summary
                        valid_results = {k: v for k, v in current_results.items() if 'error' not in v}
                        
                        summary_data = []
                        total_cost = sum([safe_get(r, 'total_cost', 0) for r in valid_results.values()])
                        
                        for env, result in valid_results.items():
                            cost_breakdown = safe_get(result, 'cost_breakdown', {})
                            result_total_cost = safe_get(result, 'total_cost', 0)
                            percentage = (result_total_cost / total_cost * 100) if total_cost > 0 else 0
                            
                            # Adapt for new structure
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
                        # Bulk cost summary
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
                            # Adapt for new structure
                            instance_type_display = safe_get_str(result, 'instance_type', 'N/A')
                            vcpus_display = safe_get(result, 'actual_vCPUs', 0)
                            ram_gb_display = safe_get(result, 'actual_RAM_GB', 0)
                            instance_cost_display = safe_get(result, 'instance_cost', 0)
                            
                            if 'writer' in result: # Multi-AZ structure
                                writer_info = result['writer']
                                instance_type_display = safe_get_str(writer_info, 'instance_type', 'N/A')
                                vcpus_display = safe_get(writer_info, 'actual_vCPUs', 0)
                                ram_gb_display = safe_get(writer_info, 'actual_RAM_GB', 0)
                                instance_cost_display = safe_get(writer_info, 'instance_cost', 0) # Writer instance cost
                                if result['readers']:
                                    for reader_info in result['readers']:
                                        instance_cost_display += safe_get(reader_info, 'instance_cost', 0) # Add reader costs
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
                        # Bulk server CSV export
                        df_data = []
                        for server_name, server_results in current_results.items():
                            if 'error' not in server_results:
                                result = server_results.get('PROD', list(server_results.values())[0])
                                if 'error' not in result:
                                    # Adapt for new structure
                                    instance_type_display = ""
                                    vcpus_display = 0
                                    ram_gb_display = 0
                                    instance_cost_display = 0

                                    if 'writer' in result: # Multi-AZ structure
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
                        'ai_insights': ai_insights,
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
                        # Single server cost summary
                        valid_results = {k: v for k, v in current_results.items() if 'error' not in v}
                        
                        summary_data = []
                        total_cost = sum([safe_get(r, 'total_cost', 0) for r in valid_results.values()])
                        
                        for env, result in valid_results.items():
                            cost_breakdown = safe_get(result, 'cost_breakdown', {})
                            result_total_cost = safe_get(result, 'total_cost', 0)
                            percentage = (result_total_cost / total_cost * 100) if total_cost > 0 else 0
                            
                            # Adapt for new structure
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
                        # Bulk cost summary
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
                                    
                                    # Adapt for new structure
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
                    total_cost = safe_get(prod_result, 'total_cost', 0) # Now total_cost is for the entire deployment if Multi-AZ
                    
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
                            instance_type_display = safe_get_str(prod_result, 'instance_type', 'N/A') # Original single instance type
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

with tab4:
    st.header("💰 Financial Analysis & Advanced Visualizations")
    
    # Determine which results to use
    current_results = None
    if st.session_state.current_analysis_mode == 'single' and st.session_state.results:
        current_results = st.session_state.results
        analysis_title = "Single Server"
    elif st.session_state.current_analysis_mode == 'bulk' and st.session_state.bulk_results:
        # For bulk, we'll analyze the aggregated results
        current_results = st.session_state.bulk_results
        analysis_title = "Bulk Analysis"
    
    if not current_results:
        st.info("💡 Generate sizing recommendations first to enable financial analysis.")
    else:
        if st.session_state.current_analysis_mode == 'single':
            # Single server financial analysis
            results = current_results
            valid_results = {k: v for k, v in results.items() if 'error' not in v}
            
            if valid_results:
                # Enhanced Financial Summary
                st.subheader(f"📊 {analysis_title} Financial Summary")
                
                prod_result = valid_results.get('PROD', list(valid_results.values())[0])
                total_cost = safe_get(prod_result, 'total_cost', 0) # Total deployment cost
                
                # Extract instance and storage costs considering new structure
                total_instance_cost = 0
                if 'writer' in prod_result:
                    total_instance_cost = safe_get(prod_result['cost_breakdown'], 'writer_monthly', 0) + \
                                          safe_get(prod_result['cost_breakdown'], 'readers_monthly', 0)
                else:
                    total_instance_cost = safe_get(prod_result, 'instance_cost', 0)
                
                total_storage_cost = safe_get(prod_result, 'storage_cost', 0) # Storage is shared for Multi-AZ
                
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
                
                # Enhanced Visualizations
                st.subheader("📈 Advanced Financial Visualizations")
                
                # Row 1: Cost Heatmap and Sunburst
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                    heatmap_fig = create_cost_heatmap(results)
                    if heatmap_fig:
                        st.plotly_chart(heatmap_fig, use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col2:
                    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                    # Create workload distribution instead for single server
                    if st.session_state.calculator and st.session_state.calculator.workload_characteristics:
                        workload_fig = create_workload_distribution_pie(st.session_state.calculator.workload_characteristics)
                        if workload_fig:
                            st.plotly_chart(workload_fig, use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                
        else:
            # Bulk financial analysis
            st.subheader(f"📊 {analysis_title} Financial Summary")
            
            # Aggregate bulk results
            total_servers = len(current_results)
            successful_servers = 0
            total_monthly_cost = 0
            total_annual_cost = 0
            server_costs = []
            
            for server_name, server_results in current_results.items():
                if 'error' not in server_results:
                    successful_servers += 1
                    # Get PROD environment or first available
                    result = server_results.get('PROD', list(server_results.values())[0])
                    if 'error' not in result:
                        monthly_cost = safe_get(result, 'total_cost', 0) # Total deployment cost
                        total_monthly_cost += monthly_cost
                        total_annual_cost += monthly_cost * 12
                        
                        # Adapt for new structure: get writer type or single instance type
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
            
            # Key AI Metrics
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
                # Display recommended writers/readers from AI
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

            # AI Analysis Details
            st.subheader("Comprehensive AI Analysis")
            st.markdown('<div class="advisory-box">', unsafe_allow_html=True)
            st.write(ai_insights.get("ai_analysis", "No detailed AI analysis available."))
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Migration Phases
            st.subheader("Recommended Migration Phases")
            if ai_insights.get("recommended_migration_phases"):
                st.markdown('<div class="phase-timeline">', unsafe_allow_html=True)
                for i, phase in enumerate(ai_insights["recommended_migration_phases"]):
                    st.markdown(f"**Phase {i+1}:** {phase}")
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.info("No specific migration phases recommended by AI.")

with tab6:
    st.header("📋 Export & Reporting")
    
    current_results = None
    if st.session_state.current_analysis_mode == 'single' and st.session_state.results:
        current_results = st.session_state.results
    elif st.session_state.current_analysis_mode == 'bulk' and st.session_state.bulk_results:
        current_results = st.session_state.bulk_results
    
    if not current_results:
        st.info("💡 Generate sizing recommendations first to enable reporting.")
    else:
        st.subheader("PDF Report Generation")
        
        st.markdown("""
        <div class="status-info">
            Generate a comprehensive PDF report including sizing recommendations, cost analysis, and AI insights.
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("📄 Generate PDF Report", type="secondary", use_container_width=True):
            with st.spinner("Generating PDF report... This may take a moment."):
                try:
                    # For PDF, we generally want to include all environments or a specific one.
                    # For simplicity, we'll pass all results if single, or aggregated if bulk.
                    # The report generator can then process them.
                    
                    # If in bulk mode, we might want to generate a summary PDF or multiple PDFs.
                    # For now, let's assume a summary PDF is generated from the PROD environment of bulk results if available,
                    # or the primary single server result.
                    
                    if st.session_state.current_analysis_mode == 'single':
                        recommendations_for_pdf = st.session_state.results
                    else: # Bulk mode
                        # For a single PDF, we might aggregate or pick a key environment (e.g., PROD from each)
                        # Or, the report generator should handle the full bulk_results.
                        # Let's pass the full bulk results and let report_generator decide how to summarize/display.
                        # The report_generator's loop over `recommendations.items()` handles multiple environments.
                        recommendations_for_pdf = st.session_state.bulk_results
                        # However, for a meaningful single report, it's often best to select a summary view.
                        # A better approach would be to generate a "summary of bulk results" PDF or allow user to select server.
                        # For now, if bulk, it will try to generate a report for *each* environment within *each* server, which might be very long.
                        # Let's refine this to make the PDF generation more sensible for bulk.
                        # For bulk, the PDF should summarize overall and maybe detail top N servers or just PROD.
                        # For this fix, let's make generate_enhanced_pdf_report understand bulk.
                        # The report_generator.py is already iterating over environments.
                        # So, if bulk_results are passed, it will try to make a section for each server's environments.
                        # This could be problematic if thousands of servers.
                        # Let's simplify: if bulk, create a single summary PDF.
                        
                        # Option 1: Generate a single summary PDF for bulk analysis
                        # This would require a new method in report_generator.py or adapting the existing one.
                        # For simplicity of this fix, let's keep the existing loop logic but be aware it might be very long.
                        # The current `report_generator.py` iterates `for env, rec in recommendations.items():`.
                        # If `recommendations` is `st.session_state.bulk_results`, then `env` will be 'ServerName', and `rec` will be {'PROD': {...}, 'QA': {...}}.
                        # This is NOT what the report generator expects. It expects `{'PROD': {...}, 'QA': {...}}`.
                        
                        # REFINEMENT: If bulk, we should make the user select which server's report they want.
                        # Or generate a high-level summary.
                        
                        # Let's make the PDF generation for bulk a summary report.
                        # This requires changes in report_generator.py to specifically handle bulk data.
                        # For now, let's make it work by only allowing PDF generation for single server analysis.
                        # A proper bulk report would require a different PDF structure.
                        st.warning("Bulk PDF report generation is under development. Please use CSV/JSON exports for bulk analysis.")
                        st.stop() # Prevent PDF generation for bulk for now.
                        
                    pdf_bytes = report_generator.generate_enhanced_pdf_report(
                        recommendations_for_pdf,
                        st.session_state.target_engine,
                        st.session_state.ai_insights
                    )
                    
                    if pdf_bytes:
                        st.download_button(
                            label="📥 Download PDF Report",
                            data=pdf_bytes,
                            file_name=f"aws_rds_migration_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                            mime="application/pdf",
                            use_container_width=True
                        )
                        st.success("✅ PDF report generated and ready for download!")
                    else:
                        st.error("❌ PDF report generation failed with no output.")
                except Exception as e:
                    st.error(f"❌ Error generating PDF report: {str(e)}")
                    st.code(traceback.format_exc())

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <h5>🚀 Enterprise AWS RDS Migration & Sizing Tool v2.0</h5>
    <p>AI-Powered Database Migration Analysis • Built for Enterprise Scale</p>
    <p>💡 For support and advanced features, contact your AWS solutions architect</p>
</div>
""", unsafe_allow_html=True)
