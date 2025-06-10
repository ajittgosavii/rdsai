import math
import json
import logging
import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict # Import asdict for serialization
from enum import Enum
import anthropic

class MigrationType(Enum):
    HOMOGENEOUS = "homogeneous"
    HETEROGENEOUS = "heterogeneous"

class EngineFamily(Enum):
    ORACLE = "oracle"
    SQLSERVER = "sqlserver"
    MYSQL = "mysql"
    POSTGRESQL = "postgresql"
    AURORA = "aurora"

@dataclass
class MigrationProfile:
    source_engine: str
    target_engine: str
    migration_type: MigrationType
    complexity_factor: float
    performance_adjustment: float
    feature_compatibility: float
    recommended_sizing_buffer: float

@dataclass
class WorkloadCharacteristics:
    cpu_utilization_pattern: str  # "steady", "bursty", "peak_hours"
    memory_usage_pattern: str
    io_pattern: str  # "read_heavy", "write_heavy", "mixed"
    connection_count: int
    transaction_volume: str  # "low", "medium", "high"
    analytical_workload: bool

class EnhancedRDSSizingCalculator:
    """Enhanced RDS sizing calculator with homogeneous/heterogeneous migration support and AI integration"""
    
    # Migration compatibility matrix
    MIGRATION_MATRIX = {
        # Homogeneous migrations (same engine family)
        ("oracle-ee", "oracle-ee"): MigrationProfile("oracle-ee", "oracle-ee", MigrationType.HOMOGENEOUS, 1.0, 1.0, 0.95, 1.1),
        ("oracle-se", "oracle-se"): MigrationProfile("oracle-se", "oracle-se", MigrationType.HOMOGENEOUS, 1.0, 1.0, 0.95, 1.1),
        ("postgres", "postgres"): MigrationProfile("postgres", "postgres", MigrationType.HOMOGENEOUS, 1.0, 1.0, 0.98, 1.05),
        ("mysql", "mysql"): MigrationProfile("mysql", "mysql", MigrationType.HOMOGENEOUS, 1.0, 1.0, 0.98, 1.05),
        ("sqlserver", "sqlserver"): MigrationProfile("sqlserver", "sqlserver", MigrationType.HOMOGENEOUS, 1.0, 1.0, 0.95, 1.1),
        
        # Aurora upgrades (considered homogeneous)
        ("mysql", "aurora-mysql"): MigrationProfile("mysql", "aurora-mysql", MigrationType.HOMOGENEOUS, 1.2, 1.1, 0.90, 1.15),
        ("postgres", "aurora-postgresql"): MigrationProfile("postgres", "aurora-postgresql", MigrationType.HOMOGENEOUS, 1.2, 1.1, 0.90, 1.15),
        
        # Heterogeneous migrations (different engine families)
        ("oracle-ee", "postgres"): MigrationProfile("oracle-ee", "postgres", MigrationType.HETEROGENEOUS, 1.5, 1.2, 0.75, 1.3),
        ("oracle-ee", "aurora-postgresql"): MigrationProfile("oracle-ee", "aurora-postgresql", MigrationType.HETEROGENEOUS, 1.4, 1.15, 0.80, 1.25),
        ("oracle-se", "postgres"): MigrationProfile("oracle-se", "postgres", MigrationType.HETEROGENEOUS, 1.4, 1.15, 0.80, 1.25),
        ("oracle-se", "aurora-postgresql"): MigrationProfile("oracle-se", "aurora-postgresql", MigrationType.HETEROGENEOUS, 1.3, 1.1, 0.85, 1.2),
        ("sqlserver", "postgres"): MigrationProfile("sqlserver", "postgres", MigrationType.HETEROGENEOUS, 1.3, 1.15, 0.80, 1.25),
        ("sqlserver", "aurora-postgresql"): MigrationProfile("sqlserver", "aurora-postgresql", MigrationType.HETEROGENEOUS, 1.25, 1.1, 0.85, 1.2),
        ("mysql", "postgres"): MigrationProfile("mysql", "postgres", MigrationType.HETEROGENEOUS, 1.2, 1.05, 0.90, 1.15),
        ("postgres", "mysql"): MigrationProfile("postgres", "mysql", MigrationType.HETEROGENEOUS, 1.2, 1.05, 0.90, 1.15),
    }
    
    # Environment profiles with enhanced characteristics
    ENV_PROFILES = {
        "PROD": {
            "cpu_multiplier": 1.0,
            "ram_multiplier": 1.0, 
            "storage_multiplier": 1.0,
            "performance_buffer": 1.25,
            "ha_multiplier": 1.5, # Base HA multiplier for Multi-AZ standby costs if calculated in total
            "backup_retention": 35,
            "min_instance_class": "m5",
            "cost_priority": 0.3,
            "availability_requirement": 99.99,
            "rto_minutes": 15,
            "rpo_minutes": 5,
            "description": "Production environment with maximum performance and availability"
        },
        "SQA": {
            "cpu_multiplier": 0.75,
            "ram_multiplier": 0.8,
            "storage_multiplier": 0.7,
            "performance_buffer": 1.15,
            "ha_multiplier": 1.2,
            "backup_retention": 14,
            "min_instance_class": "t3",
            "cost_priority": 0.5,
            "availability_requirement": 99.9,
            "rto_minutes": 60,
            "rpo_minutes": 30,
            "description": "System QA environment with production-like characteristics"
        },
        "QA": {
            "cpu_multiplier": 0.5,
            "ram_multiplier": 0.6,
            "storage_multiplier": 0.5,
            "performance_buffer": 1.1,
            "ha_multiplier": 1.0,
            "backup_retention": 7,
            "min_instance_class": "t3",
            "cost_priority": 0.7,
            "availability_requirement": 99.0,
            "rto_minutes": 120,
            "rpo_minutes": 60,
            "description": "Quality Assurance environment optimized for cost efficiency"
        },
        "DEV": {
            "cpu_multiplier": 0.25,
            "ram_multiplier": 0.35,
            "storage_multiplier": 0.3,
            "performance_buffer": 1.0,
            "ha_multiplier": 1.0,
            "backup_retention": 1,
            "min_instance_class": "t3",
            "cost_priority": 0.9,
            "availability_requirement": 95.0,
            "rto_minutes": 240,
            "rpo_minutes": 120,
            "description": "Development environment with minimal resources"
        }
    }
    
    def __init__(self, anthropic_api_key: Optional[str] = None, use_real_time_pricing: bool = True):
        self.use_real_time_pricing = use_real_time_pricing
        self.logger = logging.getLogger(__name__)
        
        # Initialize Claude AI client
        self.ai_client = None
        if anthropic_api_key:
            try:
                self.ai_client = anthropic.Anthropic(api_key=anthropic_api_key)
                self.logger.info("✅ Claude AI integration enabled")
            except Exception as e:
                self.logger.warning(f"Failed to initialize Claude AI: {e}")
        
        self.recommendations = {}
        self.ai_insights = {}
        self.migration_analysis = {}
        
    def set_migration_parameters(self, source_engine: str, target_engine: str, 
                               workload_characteristics: WorkloadCharacteristics = None):
        """Set migration parameters for sizing calculation"""
        
        # Get migration profile
        migration_key = (source_engine, target_engine)
        self.migration_profile = self.MIGRATION_MATRIX.get(
            migration_key, 
            MigrationProfile(source_engine, target_engine, MigrationType.HETEROGENEOUS, 1.3, 1.2, 0.7, 1.3)
        )
        
        self.workload_characteristics = workload_characteristics or WorkloadCharacteristics(
            cpu_utilization_pattern="steady",
            memory_usage_pattern="steady", 
            io_pattern="mixed",
            connection_count=100,
            transaction_volume="medium",
            analytical_workload=False
        )
        
        self.logger.info(f"Migration configured: {source_engine} → {target_engine} ({self.migration_profile.migration_type.value})")
        
    def calculate_enhanced_requirements(self, env: str, inputs: Dict) -> Dict:
        """
        Calculate requirements with migration-aware sizing,
        supporting single, Multi-AZ, Multi-AZ Cluster, and Aurora Global deployments.
        """
        
        profile = self.ENV_PROFILES[env]
        
        # Step 1: Base resource calculation from on-prem
        base_cpu_cores = inputs["on_prem_cores"] * (inputs["peak_cpu_percent"] / 100)
        base_ram_gb = inputs["on_prem_ram_gb"] * (inputs["peak_ram_percent"] / 100)
        
        # Step 2: Apply migration adjustments to base requirements
        if hasattr(self, 'migration_profile'):
            migration_cpu_factor = self.migration_profile.complexity_factor * self.migration_profile.performance_adjustment
            migration_ram_factor = self.migration_profile.complexity_factor
            
            if self.migration_profile.migration_type == MigrationType.HETEROGENEOUS:
                migration_cpu_factor *= 1.1
                migration_ram_factor *= 1.15
                
            base_cpu_cores *= migration_cpu_factor
            base_ram_gb *= migration_ram_factor
        
        # Step 3: Apply workload characteristics adjustments
        if hasattr(self, 'workload_characteristics'):
            workload = self.workload_characteristics
            if workload.cpu_utilization_pattern == "bursty":
                base_cpu_cores *= 1.2
            elif workload.cpu_utilization_pattern == "peak_hours":
                base_cpu_cores *= 1.15
            
            if workload.memory_usage_pattern == "high_variance":
                base_ram_gb *= 1.25
            elif workload.analytical_workload:
                base_ram_gb *= 1.5

        # Step 4: Apply environment-specific factors for the primary (writer) instance
        # The base_cpu_cores and base_ram_gb after migration/workload adjustments are effectively for the primary instance
        writer_cpu_requirement = base_cpu_cores * profile["cpu_multiplier"] * profile["performance_buffer"]
        writer_ram_requirement = base_ram_gb * profile["ram_multiplier"] * profile["performance_buffer"]

        # Apply minimums for writer
        env_minimums = {
            "PROD": {"cpu": 4, "ram": 8},
            "SQA": {"cpu": 2, "ram": 4},
            "QA": {"cpu": 2, "ram": 4}, 
            "DEV": {"cpu": 1, "ram": 2}
        }
        min_reqs = env_minimums[env]
        final_writer_cpu_requirement = max(math.ceil(writer_cpu_requirement), min_reqs["cpu"])
        final_writer_ram_requirement = max(math.ceil(writer_ram_requirement), min_reqs["ram"])
        
        # Get instance pricing data (common for all instances in this environment)
        available_instances = self._get_fallback_instance_data(inputs["region"], inputs["target_engine"])
        
        # Calculate storage (common for the cluster/deployment)
        storage_gb = self._calculate_enhanced_storage(env, profile, inputs)

        deployment_option = inputs.get("deployment", "Single-AZ")
        is_aurora_engine = inputs["target_engine"].startswith("aurora")

        if deployment_option == "Single-AZ":
            # Existing Single-AZ logic (one instance covers all)
            selected_instance = self._select_optimal_instance_enhanced(
                final_writer_cpu_requirement, final_writer_ram_requirement, env, profile, available_instances, inputs
            )
            # Calculate costs for this single instance
            costs = self._calculate_comprehensive_costs(selected_instance, storage_gb, env, profile, inputs)
            
            # Generate AI-enhanced advisories for this single instance setup
            advisories = self._generate_ai_enhanced_advisories(
                selected_instance, final_writer_cpu_requirement, final_writer_ram_requirement, env, inputs
            )

            return {
                "environment": env,
                "deployment_option": deployment_option,
                "instance_type": selected_instance["type"], # For backward compatibility in display
                "vCPUs": final_writer_cpu_requirement,
                "RAM_GB": final_writer_ram_requirement,
                "actual_vCPUs": selected_instance["vCPU"],
                "actual_RAM_GB": selected_instance["memory"],
                "storage_GB": storage_gb,
                "total_cost": costs["total_monthly"],
                "instance_cost": costs["instance_monthly"], # Instance cost for this single instance
                "storage_cost": costs["storage_monthly"],
                "backup_cost": costs["backup_monthly"],
                "advisories": advisories,
                "cost_breakdown": costs,
            }
        
        else: # Multi-AZ, Multi-AZ Cluster, Aurora Global
            # Determine writer instance
            writer_instance = self._select_optimal_instance_enhanced(
                final_writer_cpu_requirement, final_writer_ram_requirement, env, profile, available_instances, inputs
            )
            writer_costs = self._calculate_comprehensive_costs(writer_instance, storage_gb, env, profile, inputs)
            # Storage and backup costs are typically shared/counted once for the cluster/deployment,
            # so we'll assign them to the writer and aggregate later.
            writer_instance_details = {
                "instance_type": writer_instance["type"],
                "vCPUs": final_writer_cpu_requirement,
                "RAM_GB": final_writer_ram_requirement,
                "actual_vCPUs": writer_instance["vCPU"],
                "actual_RAM_GB": writer_instance["memory"],
                "instance_cost": writer_costs["instance_monthly"] # Only instance cost for writer
            }

            # Determine reader instances
            num_readers = 1 # Default for Multi-AZ (standby is not always readable)
            if deployment_option == "Multi-AZ Cluster":
                num_readers = 2 # e.g., SQL Server Multi-AZ Cluster has 2 readable secondaries
            elif is_aurora_engine:
                if deployment_option == "Multi-AZ": # For Aurora, Multi-AZ implies a cluster with a reader
                    num_readers = 1 
                elif deployment_option == "Multi-AZ Cluster" or deployment_option == "Aurora Global":
                    # Scale readers based on workload and engine characteristics for Aurora
                    if self.workload_characteristics.io_pattern == "read_heavy" or \
                       self.workload_characteristics.analytical_workload:
                        if self.workload_characteristics.transaction_volume == "very_high":
                            num_readers = 5 # More readers for very high read/analytical
                        elif self.workload_characteristics.transaction_volume == "high":
                            num_readers = 3
                        else:
                            num_readers = 2
                    else:
                        num_readers = 1 # Even for mixed/write-heavy, often one reader is beneficial

            reader_instances = []
            total_readers_cost = 0

            # Calculate reader requirements (often slightly less than writer, or same depending on read load)
            reader_cpu_requirement = max(math.ceil(base_cpu_cores * 0.75), min_reqs["cpu"]) # Readers need less CPU than writer for writes
            reader_ram_requirement = max(math.ceil(base_ram_gb * 0.8), min_reqs["ram"]) # Readers often need slightly less RAM than writer unless analytical

            # Adjust reader requirements based on workload
            if self.workload_characteristics.io_pattern == "read_heavy":
                reader_cpu_requirement = max(reader_cpu_requirement, final_writer_cpu_requirement * 0.8) # Keep readers beefy for read-heavy
                reader_ram_requirement = max(reader_ram_requirement, final_writer_ram_requirement * 0.9)
            elif self.workload_characteristics.analytical_workload:
                # Analytical needs higher RAM
                reader_ram_requirement = max(reader_ram_requirement, final_writer_ram_requirement * 1.0)
                reader_cpu_requirement = max(reader_cpu_requirement, final_writer_cpu_requirement * 0.9)


            for i in range(num_readers):
                # Select optimal instance for reader
                selected_reader_instance = self._select_optimal_instance_enhanced(
                    reader_cpu_requirement, reader_ram_requirement, env, profile, available_instances, inputs
                )
                reader_instance_cost = self._calculate_comprehensive_costs(selected_reader_instance, 0, env, profile, inputs)["instance_monthly"] # Storage is handled once for cluster
                total_readers_cost += reader_instance_cost
                
                reader_instances.append({
                    "instance_type": selected_reader_instance["type"],
                    "vCPUs": reader_cpu_requirement,
                    "RAM_GB": reader_ram_requirement,
                    "actual_vCPUs": selected_reader_instance["vCPU"],
                    "actual_RAM_GB": selected_reader_instance["memory"],
                    "instance_cost": reader_instance_cost
                })

            # Calculate total cost for the entire deployment (writer + readers + shared storage/backup/features)
            total_deployment_cost = writer_costs["instance_monthly"] + total_readers_cost + \
                                    writer_costs["storage_monthly"] + writer_costs["backup_monthly"] + \
                                    writer_costs["migration_monthly"] + writer_costs["features_monthly"] + \
                                    writer_costs["data_transfer_monthly"]
            
            # Generate AI-enhanced advisories, potentially for the combined setup
            advisories = self._generate_ai_enhanced_advisories(
                writer_instance, final_writer_cpu_requirement, final_writer_ram_requirement, env, inputs
            )

            return {
                "environment": env,
                "deployment_option": deployment_option,
                "writer": writer_instance_details,
                "readers": reader_instances,
                "storage_GB": storage_gb,
                "total_cost": total_deployment_cost, # Total cost of the entire deployment
                "instance_cost": writer_costs["instance_monthly"] + total_readers_cost, # Sum of instance costs
                "storage_cost": writer_costs["storage_monthly"], # Shared storage cost
                "backup_cost": writer_costs["backup_monthly"], # Shared backup cost
                "advisories": advisories,
                "cost_breakdown": {
                    "writer_monthly": writer_costs["instance_monthly"],
                    "readers_monthly": total_readers_cost,
                    "storage_monthly": writer_costs["storage_monthly"],
                    "backup_monthly": writer_costs["backup_monthly"],
                    "migration_monthly": writer_costs["migration_monthly"],
                    "features_monthly": writer_costs["features_monthly"],
                    "data_transfer_monthly": writer_costs["data_transfer_monthly"],
                    "total_monthly": total_deployment_cost
                },
            }
    
    def _get_fallback_instance_data(self, region: str, engine: str) -> List[Dict]:
        """Get fallback instance data when AWS API is unavailable"""
        
        fallback_data = {
            "postgres": [
                {"type": "db.t3.micro", "vCPU": 2, "memory": 1, "pricing": {"ondemand_hourly": 0.0255, "ondemand_monthly": 18.4}, "instance_family": "t3"},
                {"type": "db.t3.small", "vCPU": 2, "memory": 2, "pricing": {"ondemand_hourly": 0.051, "ondemand_monthly": 36.7}, "instance_family": "t3"},
                {"type": "db.t3.medium", "vCPU": 2, "memory": 4, "pricing": {"ondemand_hourly": 0.102, "ondemand_monthly": 73.4}, "instance_family": "t3"},
                {"type": "db.t3.large", "vCPU": 2, "memory": 8, "pricing": {"ondemand_hourly": 0.204, "ondemand_monthly": 146.9}, "instance_family": "t3"},
                {"type": "db.m5.large", "vCPU": 2, "memory": 8, "pricing": {"ondemand_hourly": 0.192, "ondemand_monthly": 138.2}, "instance_family": "m5"},
                {"type": "db.m5.xlarge", "vCPU": 4, "memory": 16, "pricing": {"ondemand_hourly": 0.384, "ondemand_monthly": 276.5}, "instance_family": "m5"},
                {"type": "db.m5.2xlarge", "vCPU": 8, "memory": 32, "pricing": {"ondemand_hourly": 0.768, "ondemand_monthly": 553.0}, "instance_family": "m5"},
                {"type": "db.m5.4xlarge", "vCPU": 16, "memory": 64, "pricing": {"ondemand_hourly": 1.536, "ondemand_monthly": 1106.0}, "instance_family": "m5"},
                {"type": "db.r5.large", "vCPU": 2, "memory": 16, "pricing": {"ondemand_hourly": 0.24, "ondemand_monthly": 172.8}, "instance_family": "r5"},
                {"type": "db.r5.xlarge", "vCPU": 4, "memory": 32, "pricing": {"ondemand_hourly": 0.48, "ondemand_monthly": 345.6}, "instance_family": "r5"},
                {"type": "db.r5.2xlarge", "vCPU": 8, "memory": 64, "pricing": {"ondemand_hourly": 0.96, "ondemand_monthly": 691.2}, "instance_family": "r5"}
            ],
            "oracle-ee": [
                {"type": "db.t3.medium", "vCPU": 2, "memory": 4, "pricing": {"ondemand_hourly": 0.272, "ondemand_monthly": 195.8}, "instance_family": "t3"},
                {"type": "db.t3.large", "vCPU": 2, "memory": 8, "pricing": {"ondemand_hourly": 0.544, "ondemand_monthly": 391.7}, "instance_family": "t3"},
                {"type": "db.m5.large", "vCPU": 2, "memory": 8, "pricing": {"ondemand_hourly": 0.475, "ondemand_monthly": 342.0}, "instance_family": "m5"},
                {"type": "db.m5.xlarge", "vCPU": 4, "memory": 16, "pricing": {"ondemand_hourly": 0.95, "ondemand_monthly": 684.0}, "instance_family": "m5"},
                {"type": "db.m5.2xlarge", "vCPU": 8, "memory": 32, "pricing": {"ondemand_hourly": 1.90, "ondemand_monthly": 1368.0}, "instance_family": "m5"},
                {"type": "db.m5.4xlarge", "vCPU": 16, "memory": 64, "pricing": {"ondemand_hourly": 3.80, "ondemand_monthly": 2736.0}, "instance_family": "m5"},
                {"type": "db.r5.large", "vCPU": 2, "memory": 16, "pricing": {"ondemand_hourly": 0.60, "ondemand_monthly": 432.0}, "instance_family": "r5"},
                {"type": "db.r5.xlarge", "vCPU": 4, "memory": 32, "pricing": {"ondemand_hourly": 1.20, "ondemand_monthly": 864.0}, "instance_family": "r5"},
                {"type": "db.r5.2xlarge", "vCPU": 8, "memory": 64, "pricing": {"ondemand_hourly": 2.40, "ondemand_monthly": 1728.0}, "instance_family": "r5"},
            ],
            "aurora-postgresql": [
                {"type": "db.t3.medium", "vCPU": 2, "memory": 4, "pricing": {"ondemand_hourly": 0.082, "ondemand_monthly": 59.0}, "instance_family": "t3"},
                {"type": "db.t4g.medium", "vCPU": 2, "memory": 4, "pricing": {"ondemand_hourly": 0.073, "ondemand_monthly": 52.6}, "instance_family": "t4g"},
                {"type": "db.r5.large", "vCPU": 2, "memory": 16, "pricing": {"ondemand_hourly": 0.285, "ondemand_monthly": 205.2}, "instance_family": "r5"},
                {"type": "db.r5.xlarge", "vCPU": 4, "memory": 32, "pricing": {"ondemand_hourly": 0.57, "ondemand_monthly": 410.4}, "instance_family": "r5"},
                {"type": "db.r5.2xlarge", "vCPU": 8, "memory": 64, "pricing": {"ondemand_hourly": 1.14, "ondemand_monthly": 820.8}, "instance_family": "r5"},
                {"type": "db.r6g.large", "vCPU": 2, "memory": 16, "pricing": {"ondemand_hourly": 0.256, "ondemand_monthly": 184.3}, "instance_family": "r6g"},
                {"type": "db.r6g.xlarge", "vCPU": 4, "memory": 32, "pricing": {"ondemand_hourly": 0.512, "ondemand_monthly": 368.6}, "instance_family": "r6g"},
                {"type": "db.serverless", "vCPU": 0, "memory": 0, "pricing": {"ondemand_hourly": 0.12, "ondemand_monthly": 86.4}, "instance_family": "serverless"}, # Placeholder, pricing for ACU differs
            ],
            "aurora-mysql": [
                {"type": "db.t3.medium", "vCPU": 2, "memory": 4, "pricing": {"ondemand_hourly": 0.082, "ondemand_monthly": 59.0}, "instance_family": "t3"},
                {"type": "db.t4g.medium", "vCPU": 2, "memory": 4, "pricing": {"ondemand_hourly": 0.073, "ondemand_monthly": 52.6}, "instance_family": "t4g"},
                {"type": "db.r5.large", "vCPU": 2, "memory": 16, "pricing": {"ondemand_hourly": 0.285, "ondemand_monthly": 205.2}, "instance_family": "r5"},
                {"type": "db.r5.xlarge", "vCPU": 4, "memory": 32, "pricing": {"ondemand_hourly": 0.57, "ondemand_monthly": 410.4}, "instance_family": "r5"},
                {"type": "db.r5.2xlarge", "vCPU": 8, "memory": 64, "pricing": {"ondemand_hourly": 1.14, "ondemand_monthly": 820.8}, "instance_family": "r5"},
                {"type": "db.serverless", "vCPU": 0, "memory": 0, "pricing": {"ondemand_hourly": 0.12, "ondemand_monthly": 86.4}, "instance_family": "serverless"}, # Placeholder, pricing for ACU differs
            ],
            "mysql": [
                {"type": "db.t3.micro", "vCPU": 2, "memory": 1, "pricing": {"ondemand_hourly": 0.020, "ondemand_monthly": 14.4}, "instance_family": "t3"},
                {"type": "db.t3.small", "vCPU": 2, "memory": 2, "pricing": {"ondemand_hourly": 0.040, "ondemand_monthly": 28.8}, "instance_family": "t3"},
                {"type": "db.t3.medium", "vCPU": 2, "memory": 4, "pricing": {"ondemand_hourly": 0.080, "ondemand_monthly": 57.6}, "instance_family": "t3"},
                {"type": "db.m5.large", "vCPU": 2, "memory": 8, "pricing": {"ondemand_hourly": 0.150, "ondemand_monthly": 108.0}, "instance_family": "m5"},
                {"type": "db.m5.xlarge", "vCPU": 4, "memory": 16, "pricing": {"ondemand_hourly": 0.300, "ondemand_monthly": 216.0}, "instance_family": "m5"},
                {"type": "db.r5.large", "vCPU": 2, "memory": 16, "pricing": {"ondemand_hourly": 0.188, "ondemand_monthly": 135.4}, "instance_family": "r5"},
            ]
        }
        
        # Apply regional pricing adjustments
        regional_multipliers = {
            "us-east-1": 1.0, "us-west-1": 1.08, "us-west-2": 1.08,
            "eu-west-1": 1.15, "eu-central-1": 1.18,
            "ap-southeast-1": 1.20, "ap-northeast-1": 1.25
        }
        
        multiplier = regional_multipliers.get(region, 1.1)
        engine_data = fallback_data.get(engine, fallback_data["postgres"])
        
        # Apply regional adjustment
        adjusted_data = []
        for instance in engine_data:
            adjusted_instance = instance.copy()
            adjusted_instance["pricing"] = {
                "ondemand_hourly": instance["pricing"]["ondemand_hourly"] * multiplier,
                "ondemand_monthly": instance["pricing"]["ondemand_monthly"] * multiplier
            }
            adjusted_data.append(adjusted_instance)
        
        return adjusted_data
    
    def _select_optimal_instance_enhanced(self, cpu_req: int, ram_req: int, env: str, 
                                        profile: Dict, available_instances: List[Dict], inputs: Dict) -> Dict:
        """Enhanced instance selection with migration and workload awareness"""
        
        if not available_instances:
            raise ValueError(f"No instances available for {inputs['target_engine']} in {inputs['region']}")
        
        # Filter instances that meet minimum requirements
        suitable_instances = []
        for instance in available_instances:
            meets_cpu = instance["vCPU"] >= cpu_req
            meets_ram = instance["memory"] >= ram_req
            
            # Check instance family restrictions
            min_family = profile["min_instance_class"]
            instance_family = instance.get("instance_family", instance["type"].split('.')[1] if '.' in instance["type"] else "unknown")
            
            family_scores = {"t3": 1, "t4g": 1, "m5": 2, "m6i": 2, "r5": 3, "r6g": 3, "c5": 2, "serverless": 0} # Added serverless
            min_score = family_scores.get(min_family, 1)
            instance_score = family_scores.get(instance_family, 1)
            
            # Allow serverless only if target engine is Aurora and deployment option supports it
            if instance_family == "serverless":
                if inputs["target_engine"].startswith("aurora") and \
                   (inputs.get("deployment") == "Serverless" or inputs.get("deployment") == "Single-AZ"): # Simplified for serverless
                    meets_family = True
                else:
                    meets_family = False
            else:
                meets_family = instance_score >= min_score

            if meets_cpu and meets_ram and meets_family:
                suitable_instances.append(instance)
        
        # Apply tolerance for non-production environments if no exact matches
        if not suitable_instances and env != "PROD":
            tolerance_levels = {"DEV": 0.7, "QA": 0.8, "SQA": 0.9}
            tolerance = tolerance_levels.get(env, 0.8)
            
            for instance in available_instances:
                # Ensure serverless is not picked unless explicitly for Aurora Serverless deployment
                if instance.get("instance_family") == "serverless":
                    continue 

                if (instance["vCPU"] >= cpu_req * tolerance and 
                    instance["memory"] >= ram_req * tolerance):
                    suitable_instances.append(instance)
        
        if not suitable_instances:
            # Last resort: return the largest available instance that is not serverless,
            # or serverless if it's the only one and Aurora is target.
            non_serverless_instances = [i for i in available_instances if i.get("instance_family") != "serverless"]
            if non_serverless_instances:
                suitable_instances = [max(non_serverless_instances, key=lambda x: x["vCPU"] + x["memory"])]
            elif inputs["target_engine"].startswith("aurora"):
                # If only serverless instances are available for Aurora, pick it as a last resort
                suitable_instances = [i for i in available_instances if i.get("instance_family") == "serverless"]
                if not suitable_instances: # still no suitable, raise error
                    raise ValueError(f"No suitable instances found for {inputs['target_engine']} with requirements CPU:{cpu_req}, RAM:{ram_req}")
            else:
                raise ValueError(f"No suitable instances found for {inputs['target_engine']} with requirements CPU:{cpu_req}, RAM:{ram_req}")

        # Advanced scoring algorithm
        def calculate_enhanced_score(instance):
            # If serverless, give a base score as vCPU/memory are 0 for direct sizing
            if instance.get("instance_family") == "serverless":
                 # Serverless scoring needs to be different as it doesn't have fixed vCPU/memory
                 # Prioritize based on target engine suitability and cost-effectiveness for bursts
                 if inputs["target_engine"].startswith("aurora"):
                     return 1000 / (instance["pricing"]["ondemand_hourly"] + 0.1) # Prioritize lower cost per ACU
                 return 0 # Not suitable for other engines
            
            # Resource efficiency
            cpu_ratio = instance["vCPU"] / max(cpu_req, 1)
            ram_ratio = instance["memory"] / max(ram_req, 1)
            
            # Penalize over-provisioning
            cpu_waste = max(0, cpu_ratio - 1.5) * 0.3
            ram_waste = max(0, ram_ratio - 1.5) * 0.3
            waste_penalty = cpu_waste + ram_waste
            
            # Cost factor
            base_cost_score = 1000 / (instance["pricing"]["ondemand_hourly"] + 1)
            
            # Migration-specific adjustments
            migration_bonus = 0
            if hasattr(self, 'migration_profile'):
                if self.migration_profile.migration_type == MigrationType.HETEROGENEOUS:
                    # Prefer more capable instances for heterogeneous migrations
                    instance_family = instance.get("instance_family", "unknown")
                    if instance_family in ["m5", "m6i", "r5", "r6g"]:
                        migration_bonus = 0.2
                elif self.migration_profile.migration_type == MigrationType.HOMOGENEOUS:
                    # Can be more aggressive with cost optimization
                    if env in ["DEV", "QA"] and instance_family in ["t3", "t4g"]:
                        migration_bonus = 0.15
            
            # Workload-specific adjustments
            workload_bonus = 0
            if hasattr(self, 'workload_characteristics'):
                workload = self.workload_characteristics
                if workload.io_pattern == "read_heavy" and "r5" in instance["type"]:
                    workload_bonus += 0.1
                if workload.analytical_workload and instance["memory"] / instance["vCPU"] >= 8:
                    workload_bonus += 0.15
                if workload.cpu_utilization_pattern == "bursty" and instance_family in ["t3", "t4g"]:
                    workload_bonus += 0.1
            
            # Environment-specific scoring
            cost_priority = profile["cost_priority"]
            performance_priority = 1 - cost_priority
            
            efficiency_score = (2.0 - waste_penalty) * performance_priority
            cost_score = base_cost_score * cost_priority * 0.001  # Normalize cost score
            
            total_score = efficiency_score + cost_score + migration_bonus + workload_bonus
            
            return total_score
        
        # Select the best instance
        best_instance = max(suitable_instances, key=calculate_enhanced_score)
        
        self.logger.info(f"Selected {best_instance['type']} for {env} environment")
        return best_instance
    
    def _calculate_enhanced_storage(self, env: str, profile: Dict, inputs: Dict) -> int:
        """Calculate storage requirements with migration considerations"""
        
        base_storage = inputs["storage_current_gb"]
        growth_factor = (1 + inputs["storage_growth_rate"]) ** inputs.get("years", 3)
        projected_storage = base_storage * growth_factor
        
        # Apply environment multiplier
        env_storage = projected_storage * profile["storage_multiplier"]
        
        # Migration-specific adjustments
        if hasattr(self, 'migration_profile'):
            if self.migration_profile.migration_type == MigrationType.HETEROGENEOUS:
                # Need extra space for migration process and potential schema changes
                env_storage *= 1.4
            else:
                # Homogeneous migrations need less extra space
                env_storage *= 1.2
        
        # Add environment-specific buffer
        buffer_factors = {"PROD": 1.5, "SQA": 1.3, "QA": 1.2, "DEV": 1.1}
        buffer = buffer_factors.get(env, 1.2)
        
        final_storage = env_storage * buffer
        
        # Apply minimums
        min_storage = {"PROD": 100, "SQA": 50, "QA": 50, "DEV": 20}[env]
        
        return max(min_storage, math.ceil(final_storage))
    
    def _calculate_comprehensive_costs(self, instance: Dict, storage_gb: int, 
                                     env: str, profile: Dict, inputs: Dict) -> Dict:
        """
        Calculate comprehensive costs for a SINGLE instance, plus shared storage/backup/features.
        deployment_factor is used to estimate cost for Multi-AZ/Aurora for the single instance type
        if not explicitly broken down into writer/reader.
        """
        
        # Instance costs
        hourly_rate = instance["pricing"]["ondemand_hourly"]
        
        # Deployment factor is now handled by the caller (calculate_enhanced_requirements)
        # It's only used here if we are calculating for a "Single-AZ" equivalent setup or for the writer/reader base cost.
        # For Aurora Serverless, vCPU and memory are 0, so hourly_rate directly reflects ACU cost per hour.
        if instance.get("instance_family") == "serverless":
            monthly_instance = hourly_rate * 24 * 30 # ACU cost is per hour
        else:
            monthly_instance = hourly_rate * 24 * 30 
        
        # Storage costs (only applied once for the entire deployment/cluster, usually to the writer)
        monthly_storage = 0
        if storage_gb > 0: # Only calculate if storage is passed (i.e., for the primary instance)
            storage_type = inputs.get("storage_type", "gp3")
            
            if inputs["target_engine"].startswith("aurora"):
                storage_rate = 0.10  # Aurora storage rate per GB-month
            else:
                storage_rates = {"gp2": 0.10, "gp3": 0.08, "io1": 0.125, "io2": 0.125} # Per GB-month
                storage_rate = storage_rates.get(storage_type, 0.10)
            
            monthly_storage = storage_gb * storage_rate
        
        # Backup costs (only applied once for the entire deployment/cluster)
        monthly_backup = 0
        if storage_gb > 0: # Only calculate if storage is passed
            backup_retention_days = profile["backup_retention"]
            if inputs["target_engine"].startswith("aurora"):
                backup_rate = 0.021  # Aurora backup rate per GB-month
            else:
                backup_rate = 0.095  # RDS backup rate per GB-month
            
            monthly_backup = storage_gb * backup_rate * (backup_retention_days / 30) # Assuming daily backups based on storage used
        
        # Migration-specific costs (applied once for the entire deployment)
        migration_costs = 0
        if hasattr(self, 'migration_profile'):
            if self.migration_profile.migration_type == MigrationType.HETEROGENEOUS:
                # Additional costs for DMS, testing, validation. Apply to primary instance cost.
                migration_costs = monthly_instance * 0.15 
        
        # Additional feature costs (applied once for the entire deployment)
        features_cost = 0
        # These features are typically per cluster/DB instance, not per node in Multi-AZ
        if inputs.get("enable_perf_insights", True):
            features_cost += monthly_instance * 0.1 # This needs to be refined for Multi-AZ
        if inputs.get("enable_encryption", True):
            features_cost += monthly_instance * 0.02
        if inputs.get("enable_enhanced_monitoring", True):
            features_cost += 15  # $15/month for enhanced monitoring
        
        # Data transfer costs (applied once for the entire deployment)
        data_transfer_gb = inputs.get("monthly_data_transfer_gb", 100)
        data_transfer_cost = data_transfer_gb * 0.09 # Typical outbound data transfer rate
        
        total_monthly = (monthly_instance + monthly_storage + monthly_backup + 
                        migration_costs + features_cost + data_transfer_cost)
        
        return {
            "instance_monthly": monthly_instance,
            "storage_monthly": monthly_storage,
            "backup_monthly": monthly_backup,
            "migration_monthly": migration_costs,
            "features_monthly": features_cost,
            "data_transfer_monthly": data_transfer_cost,
            "total_monthly": total_monthly, # This is the total for THIS instance type plus its share of shared costs
        }
    
    async def generate_ai_insights(self, recommendations: Dict, inputs: Dict) -> Dict:
        """Generate AI-powered insights and recommendations"""
        
        if not self.ai_client:
            return {"error": "Claude AI not available"}
        
        try:
            # Prepare context for AI analysis
            # Use asdict to serialize MigrationProfile object
            migration_profile_data = asdict(self.migration_profile) if hasattr(self, 'migration_profile') and self.migration_profile else {}
            
            # Explicitly convert MigrationType enum to its string value if present
            if 'migration_type' in migration_profile_data and isinstance(migration_profile_data['migration_type'], Enum):
                migration_profile_data['migration_type'] = migration_profile_data['migration_type'].value

            context = {
                "migration_profile_details": migration_profile_data, # Use serialized data
                "source_engine": inputs.get("source_engine", "unknown"),
                "target_engine": inputs.get("target_engine", "unknown"),
                "workload_characteristics": asdict(self.workload_characteristics) if hasattr(self, 'workload_characteristics') else {},
                "recommendations": recommendations,
                "total_cost_range": {
                    "min": min([r.get("total_cost", 0) for r in recommendations.values() if isinstance(r, dict) and "total_cost" in r]),
                    "max": max([r.get("total_cost", 0) for r in recommendations.values() if isinstance(r, dict) and "total_cost" in r])
                },
                "deployment_option": inputs.get("deployment", "Single-AZ")
            }
            
            # Adjust prompt to use the new key for migration profile details
            prompt = f"""
            As an AWS database migration expert, analyze the following RDS sizing recommendations and provide insights:

            Migration Context:
            - Migration Type: {context['migration_profile_details'].get('migration_type', 'unknown')}
            - Source Engine: {context['source_engine']}
            - Target Engine: {context['target_engine']}
            - Monthly Cost Range: ${context['total_cost_range']['min']:,.2f} - ${context['total_cost_range']['max']:,.2f}
            - Deployment Option: {context['deployment_option']}
            - Workload Characteristics: {json.dumps(context['workload_characteristics'], indent=2)}
            - Detailed Migration Profile: {json.dumps(context['migration_profile_details'], indent=2)}

            Please provide:
            1. Migration risk assessment and mitigation strategies
            2. Cost optimization opportunities
            3. Performance optimization recommendations
            4. Alternative architecture suggestions (e.g., number of writers/readers for Multi-AZ setups, considering deployment option and workload characteristics)
            5. Timeline and migration approach recommendations

            Focus on actionable insights that will help ensure a successful migration.
            """
            
            response = self.ai_client.messages.create(
                model="claude-3-5-sonnet-20240620", # Changed model here
                max_tokens=2000,
                messages=[{"role": "user", "content": prompt}]
            )
            
            insights = {
                "ai_analysis": response.content[0].text,
                "risk_level": self._assess_migration_risk(),
                "cost_optimization_potential": self._estimate_cost_optimization(),
                "recommended_migration_phases": self._suggest_migration_phases()
            }

            # AI-driven writer/reader recommendations based on deployment option and workload
            # This logic should be placed *after* the AI response from Claude if you want Claude
            # to determine it, or here if it's a rule-based recommendation by the tool.
            # For now, keeping it rule-based as per previous request for clarity and determinism.
            recommended_writers = 1
            recommended_readers = 0 # Default for Single-AZ

            if context['deployment_option'] == 'Multi-AZ':
                recommended_readers = 1 # Standby for Multi-AZ
            elif context['deployment_option'] == 'Multi-AZ Cluster':
                recommended_readers = 2 # Typically 1 primary, 2 readable secondaries for SQL Server/MySQL Multi-AZ Clusters
                # For Aurora, it could be more readers based on workload
                if context['target_engine'].startswith('aurora'):
                    if context['workload_characteristics'].get('io_pattern') == 'read_heavy' or \
                       context['workload_characteristics'].get('analytical_workload'):
                        # Scale based on transaction volume for read-heavy/analytical Aurora
                        if context['workload_characteristics'].get('transaction_volume') == 'very_high':
                            recommended_readers = 5
                        elif context['workload_characteristics'].get('transaction_volume') == 'high':
                            recommended_readers = 4
                        else:
                            recommended_readers = 3 
                    else:
                        recommended_readers = 2
            elif context['deployment_option'] == 'Aurora Global':
                recommended_readers = 3 # Typically more readers for global deployments
                if context['workload_characteristics'].get('io_pattern') == 'read_heavy' or \
                   context['workload_characteristics'].get('analytical_workload'):
                    if context['workload_characteristics'].get('transaction_volume') == 'very_high':
                        recommended_readers = 7
                    elif context['workload_characteristics'].get('transaction_volume') == 'high':
                        recommended_readers = 5
                    else:
                        recommended_readers = 4
                else:
                    recommended_readers = 3
            
            insights['recommended_writers'] = recommended_writers
            insights['recommended_readers'] = recommended_readers

            return insights
            
        except Exception as e:
            self.logger.error(f"Error generating AI insights: {e}")
            return {"error": str(e)}
    
    def _generate_ai_enhanced_advisories(self, instance: Dict, cpu_req: int, ram_req: int, 
                                       env: str, inputs: Dict) -> List[str]:
        """Generate AI-enhanced advisories"""
        
        advisories = []
        
        # Standard advisories
        # Ensure 'instance_type' exists before trying to access it for ratio calculation.
        # If the input instance is a 'writer' or 'reader' dict in a Multi-AZ context,
        # it might not have 'vCPU' and 'memory' directly, but rather 'actual_vCPUs' and 'actual_RAM_GB'.
        # Assuming the 'instance' dict passed here is the actual selected instance with 'vCPU' and 'memory'.
        if 'vCPU' in instance and 'memory' in instance:
            cpu_ratio = instance["vCPU"] / max(cpu_req, 1)
            ram_ratio = instance["memory"] / max(ram_req, 1)
            
            if cpu_ratio > 2:
                advisories.append(f"⚠️ CPU over-provisioned: {instance['vCPU']} vCPUs vs {cpu_req} required. Consider smaller instance or Reserved Instance for cost savings.")
            
            if ram_ratio > 2:
                advisories.append(f"⚠️ RAM over-provisioned: {instance['memory']}GB vs {ram_req}GB required. Memory-optimized instances may not be necessary.")
        
        # Migration-specific advisories
        if hasattr(self, 'migration_profile'):
            if self.migration_profile.migration_type == MigrationType.HETEROGENEOUS:
                advisories.append(f"🔄 Heterogeneous migration detected. Consider using AWS DMS and AWS SCT for schema conversion.")
                advisories.append(f"⏱️ Plan for extended testing phase due to {(1-self.migration_profile.feature_compatibility)*100:.0f}% feature compatibility gap.")
                
                if self.migration_profile.source_engine.startswith("oracle") and inputs["target_engine"].startswith("postgres"):
                    advisories.append("🎯 Oracle→PostgreSQL: Review stored procedures, packages, and Oracle-specific functions for conversion.")
                    
            else:
                advisories.append("✅ Homogeneous migration should have minimal compatibility issues.")
        
        # Environment-specific advisories
        if env == "PROD":
            if inputs.get("deployment", "Multi-AZ") == "Single-AZ":
                advisories.append("🚨 Production environment should use Multi-AZ deployment for high availability.")
            if not inputs.get("enable_encryption", True):
                advisories.append("🔒 Enable encryption at rest for production compliance.")
                
        elif env in ["DEV", "QA"]:
            # Check for serverless target engine if applicable
            if inputs["target_engine"].startswith("aurora") and inputs.get("deployment") == "Serverless":
                 advisories.append("💡 Aurora Serverless is selected for dev/test. Monitor ACU usage for cost efficiency.")
            elif instance.get("pricing", {}).get("ondemand_hourly", 0) > 1.0: # Check the hourly cost of the instance
                advisories.append("💡 Consider Aurora Serverless or smaller instances for variable dev/test workloads.")
            advisories.append("⏰ Consider scheduling start/stop for non-production environments to reduce costs.")
        
        # Workload-specific advisories
        if hasattr(self, 'workload_characteristics'):
            workload = self.workload_characteristics
            if workload.analytical_workload and not instance["type"].startswith("db.r"):
                advisories.append("📊 Analytical workloads may benefit from memory-optimized instances (R5/R6g family).")
            
            # This advisory should be contextually aware: if Multi-AZ with readers already, don't suggest more
            if workload.io_pattern == "read_heavy" and \
               inputs.get("deployment") in ["Single-AZ", "Multi-AZ"]: # Suggest if not already a multi-reader setup
                advisories.append("📖 Read-heavy workload: Consider read replicas for better performance and cost optimization.")
        
        return advisories
    
    def _assess_migration_risk(self) -> str:
        """Assess overall migration risk level"""
        if not hasattr(self, 'migration_profile'):
            return "MEDIUM"
        
        risk_factors = []
        
        # Migration type risk
        if self.migration_profile.migration_type == MigrationType.HETEROGENEOUS:
            risk_factors.append(0.6)
        else:
            risk_factors.append(0.2)
        
        # Feature compatibility risk
        compatibility_risk = 1 - self.migration_profile.feature_compatibility
        risk_factors.append(compatibility_risk)
        
        # Complexity risk
        complexity_risk = (self.migration_profile.complexity_factor - 1) / 2
        risk_factors.append(complexity_risk)
        
        overall_risk = sum(risk_factors) / len(risk_factors)
        
        if overall_risk > 0.6:
            return "HIGH"
        elif overall_risk > 0.3:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _estimate_cost_optimization(self) -> float:
        """Estimate potential cost optimization percentage"""
        optimization_potential = 0.0
        
        # Reserved Instance savings (20-40%)
        optimization_potential += 0.25
        
        # Storage optimization (10-30%)
        optimization_potential += 0.15
        
        # Instance rightsizing (5-25%)
        optimization_potential += 0.12
        
        # Environment scheduling for non-prod (40-70% for dev/test)
        optimization_potential += 0.20
        
        return min(optimization_potential, 0.45)  # Cap at 45%
    
    def _suggest_migration_phases(self) -> List[str]:
        """Suggest migration phases based on migration type and complexity"""
        
        if not hasattr(self, 'migration_profile'):
            return ["Phase 1: Assessment", "Phase 2: Migration", "Phase 3: Optimization"]
        
        if self.migration_profile.migration_type == MigrationType.HETEROGENEOUS:
            return [
                "Phase 1: Assessment & Schema Analysis (2-4 weeks)",
                "Phase 2: AWS SCT Schema Conversion (1-2 weeks)", 
                "Phase 3: DMS Setup & Initial Data Migration (1-2 weeks)",
                "Phase 4: Application Code Conversion (4-8 weeks)",
                "Phase 5: Testing & Validation (2-4 weeks)",
                "Phase 6: Cutover & Go-Live (1 week)",
                "Phase 7: Post-Migration Optimization (2-4 weeks)"
            ]
        else:
            return [
                "Phase 1: Assessment & Planning (1-2 weeks)",
                "Phase 2: DMS Setup & Testing (1 week)",
                "Phase 3: Data Migration (1-2 weeks)", 
                "Phase 4: Application Testing (1-2 weeks)",
                "Phase 5: Cutover & Go-Live (1 week)",
                "Phase 6: Performance Optimization (1-2 weeks)"
            ]
    
    def generate_comprehensive_recommendations(self, inputs: Dict) -> Dict:
        """Generate comprehensive recommendations for all environments"""
        
        self.logger.info(f"Generating recommendations for {inputs['target_engine']} migration")
        
        recommendations = {}
        for env in self.ENV_PROFILES:
            try:
                # Pass all inputs, including deployment_option
                recommendations[env] = self.calculate_enhanced_requirements(env, inputs)
            except Exception as e:
                self.logger.error(f"Error calculating {env} environment: {e}")
                recommendations[env] = {"error": str(e)}
        
        return recommendations

