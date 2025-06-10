import boto3
import json
import time
import logging
from datetime import datetime, timedelta
from botocore.exceptions import ClientError, NoCredentialsError
from typing import Dict, List, Optional, Tuple
import concurrent.futures
from functools import lru_cache

class EnhancedAWSPricingAPI:
    """Enhanced AWS Pricing API with caching and multi-region support"""
    
    SUPPORTED_REGIONS = [
        "us-east-1", "us-west-1", "us-west-2", "eu-west-1", "eu-central-1",
        "ap-southeast-1", "ap-southeast-2", "ap-northeast-1", "ca-central-1"
    ]
    
    CACHE_DURATION = 3600  # 1 hour cache
    MAX_WORKERS = 5  # For parallel pricing requests
    
    def __init__(self, enable_caching: bool = True):
        self.enable_caching = enable_caching
        self.cache = {}
        self.last_updated = {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize AWS clients
        self.pricing_client = None
        self.rds_client = None
        self.available = self._initialize_clients()
        
    def _initialize_clients(self) -> bool:
        """Initialize AWS clients with error handling"""
        try:
            # Initialize pricing client (only available in us-east-1)
            self.pricing_client = boto3.client('pricing', region_name='us-east-1')
            
            # Test the connection
            self.pricing_client.describe_services(ServiceCode='AmazonRDS', MaxResults=1)
            
            # Initialize RDS clients for each region
            self.rds_clients = {}
            for region in self.SUPPORTED_REGIONS:
                try:
                    self.rds_clients[region] = boto3.client('rds', region_name=region)
                except Exception as e:
                    self.logger.warning(f"Failed to initialize RDS client for {region}: {e}")
            
            self.logger.info("✅ AWS Pricing API initialized successfully")
            return True
            
        except (NoCredentialsError, ClientError) as e:
            self.logger.warning(f"AWS Pricing API not available: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Unexpected error initializing AWS clients: {e}")
            return False
    
    def get_rds_instance_pricing(self, region: str, engine: str, instance_types: List[str] = None) -> Dict:
        """Get RDS instance pricing with enhanced filtering and caching"""
        cache_key = f"rds_{region}_{engine}_{hash(tuple(instance_types or []))}"
        
        # Check cache
        if self._is_cache_valid(cache_key):
            self.logger.info(f"Using cached pricing for {engine} in {region}")
            return self.cache[cache_key]
        
        if not self.available:
            return self._get_fallback_rds_pricing(region, engine)
        
        try:
            pricing_data = self._fetch_rds_pricing_data(region, engine, instance_types)
            
            # Cache the results
            if self.enable_caching:
                self.cache[cache_key] = pricing_data
                self.last_updated[cache_key] = time.time()
            
            return pricing_data
            
        except Exception as e:
            self.logger.error(f"Error fetching RDS pricing: {e}")
            return self._get_fallback_rds_pricing(region, engine)
    
    def _fetch_rds_pricing_data(self, region: str, engine: str, instance_types: List[str] = None) -> Dict:
        """Fetch real-time RDS pricing data from AWS API"""
        
        # Map engines to AWS API format
        engine_mapping = {
            'mysql': 'MySQL',
            'postgres': 'PostgreSQL', 
            'oracle-ee': 'Oracle',
            'oracle-se': 'Oracle',
            'oracle-se1': 'Oracle',
            'oracle-se2': 'Oracle',
            'sqlserver-ee': 'SQL Server',
            'sqlserver-se': 'SQL Server',
            'sqlserver-ex': 'SQL Server',
            'sqlserver-web': 'SQL Server',
            'aurora-mysql': 'Aurora MySQL',
            'aurora-postgresql': 'Aurora PostgreSQL',
            'mariadb': 'MariaDB'
        }
        
        aws_engine = engine_mapping.get(engine, 'PostgreSQL')
        
        # Build filters
        filters = [
            {'Type': 'TERM_MATCH', 'Field': 'regionCode', 'Value': region},
            {'Type': 'TERM_MATCH', 'Field': 'databaseEngine', 'Value': aws_engine},
            {'Type': 'TERM_MATCH', 'Field': 'deploymentOption', 'Value': 'Single-AZ'}
        ]
        
        # Add engine-specific filters
        if engine.startswith('oracle'):
            edition_map = {
                'oracle-ee': 'Enterprise',
                'oracle-se': 'Standard',
                'oracle-se1': 'Standard One',
                'oracle-se2': 'Standard Two'
            }
            edition = edition_map.get(engine, 'Standard')
            filters.append({'Type': 'TERM_MATCH', 'Field': 'databaseEdition', 'Value': edition})
        
        elif engine.startswith('sqlserver'):
            edition_map = {
                'sqlserver-ee': 'Enterprise',
                'sqlserver-se': 'Standard',
                'sqlserver-ex': 'Express',
                'sqlserver-web': 'Web'
            }
            edition = edition_map.get(engine, 'Standard')
            filters.append({'Type': 'TERM_MATCH', 'Field': 'databaseEdition', 'Value': edition})
        
        # Fetch pricing data
        instances = {}
        next_token = None
        max_results = 100
        
        while len(instances) < max_results:
            try:
                params = {
                    'ServiceCode': 'AmazonRDS',
                    'Filters': filters,
                    'MaxResults': 20
                }
                
                if next_token:
                    params['NextToken'] = next_token
                
                response = self.pricing_client.get_products(**params)
                
                for price_item in response['PriceList']:
                    try:
                        product = json.loads(price_item)
                        attributes = product['product']['attributes']
                        instance_type = attributes.get('instanceType')
                        
                        if not instance_type or not instance_type.startswith('db.'):
                            continue
                        
                        # Filter by specific instance types if provided
                        if instance_types and instance_type not in instance_types:
                            continue
                        
                        # Extract pricing
                        terms = product['terms']['OnDemand']
                        price_dimension = next(iter(terms.values()))['priceDimensions']
                        hourly_price = next(iter(price_dimension.values()))['pricePerUnit']['USD']
                        
                        # Build instance data
                        instance_data = {
                            "type": instance_type,
                            "vCPU": int(attributes.get('vcpu', '0')),
                            "memory": self._parse_memory(attributes.get('memory', '0 GiB')),
                            "storage_type": attributes.get('storageType', 'EBS'),
                            "network_performance": attributes.get('networkPerformance', 'Unknown'),
                            "max_iops": int(attributes.get('maxIops', '0')),
                            "pricing": {
                                "ondemand_hourly": float(hourly_price),
                                "ondemand_monthly": float(hourly_price) * 24 * 30
                            },
                            "instance_family": instance_type.split('.')[1] if '.' in instance_type else 'unknown',
                            "enhanced_networking": attributes.get('enhancedNetworkingSupported', 'No') == 'Yes',
                            "processor_features": attributes.get('processorFeatures', {})
                        }
                        
                        instances[instance_type] = instance_data
                        
                    except (KeyError, ValueError, TypeError) as e:
                        self.logger.debug(f"Error parsing pricing item: {e}")
                        continue
                
                next_token = response.get('NextToken')
                if not next_token:
                    break
                    
            except Exception as e:
                self.logger.error(f"Error in pricing API call: {e}")
                break
        
        self.logger.info(f"Fetched pricing for {len(instances)} instances")
        return instances
    
    def get_storage_pricing(self, region: str) -> Dict:
        """Get EBS and Aurora storage pricing"""
        cache_key = f"storage_{region}"
        
        if self._is_cache_valid(cache_key):
            return self.cache[cache_key]
        
        # Enhanced storage pricing data
        storage_pricing = {
            "ebs": {
                "gp2": {"per_gb_month": 0.10, "iops_included": 3, "max_iops": 16000},
                "gp3": {"per_gb_month": 0.08, "baseline_iops": 3000, "per_iops_month": 0.005, "per_throughput_mbps": 0.04},
                "io1": {"per_gb_month": 0.125, "per_iops_month": 0.065, "max_iops": 64000},
                "io2": {"per_gb_month": 0.125, "per_iops_month": 0.065, "max_iops": 256000},
                "st1": {"per_gb_month": 0.045, "throughput_optimized": True},
                "sc1": {"per_gb_month": 0.025, "cold_storage": True}
            },
            "aurora": {
                "storage": {"per_gb_month": 0.10, "backup_included": True},
                "io": {"per_million_requests": 0.20},
                "backup": {"per_gb_month": 0.021}
            },
            "backup": {
                "rds": {"per_gb_month": 0.095},
                "aurora": {"per_gb_month": 0.021}
            }
        }
        
        # Apply regional pricing adjustments
        regional_multipliers = {
            "us-east-1": 1.0,
            "us-west-1": 1.08,
            "us-west-2": 1.08,
            "eu-west-1": 1.15,
            "eu-central-1": 1.18,
            "ap-southeast-1": 1.20,
            "ap-northeast-1": 1.25,
            "ca-central-1": 1.12
        }
        
        multiplier = regional_multipliers.get(region, 1.1)
        
        # Apply regional adjustments
        adjusted_pricing = {}
        for category, subcategory in storage_pricing.items():
            adjusted_pricing[category] = {}
            for storage_type, pricing in subcategory.items():
                adjusted_pricing[category][storage_type] = {}
                for metric, price in pricing.items():
                    if isinstance(price, (int, float)) and "per_" in metric:
                        adjusted_pricing[category][storage_type][metric] = price * multiplier
                    else:
                        adjusted_pricing[category][storage_type][metric] = price
        
        # Cache results
        if self.enable_caching:
            self.cache[cache_key] = adjusted_pricing
            self.last_updated[cache_key] = time.time()
        
        return adjusted_pricing
    
    def get_multi_region_pricing(self, engine: str, regions: List[str] = None) -> Dict:
        """Get pricing across multiple regions for comparison"""
        if not regions:
            regions = self.SUPPORTED_REGIONS[:5]  # Default to first 5 regions
        
        results = {}
        
        # Use thread pool for parallel requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.MAX_WORKERS) as executor:
            future_to_region = {
                executor.submit(self.get_rds_instance_pricing, region, engine): region 
                for region in regions
            }
            
            for future in concurrent.futures.as_completed(future_to_region):
                region = future_to_region[future]
                try:
                    results[region] = future.result()
                except Exception as e:
                    self.logger.error(f"Error fetching pricing for {region}: {e}")
                    results[region] = self._get_fallback_rds_pricing(region, engine)
        
        return results
    
    def get_reserved_instance_pricing(self, region: str, engine: str, instance_type: str, 
                                    term: str = "1yr", payment_option: str = "No Upfront") -> Dict:
        """Get Reserved Instance pricing (simplified)"""
        
        # RI discount factors (approximate)
        ri_discounts = {
            "1yr": {"No Upfront": 0.24, "Partial Upfront": 0.26, "All Upfront": 0.28},
            "3yr": {"No Upfront": 0.35, "Partial Upfront": 0.38, "All Upfront": 0.42}
        }
        
        # Get on-demand pricing first
        on_demand_pricing = self.get_rds_instance_pricing(region, engine)
        
        if instance_type not in on_demand_pricing:
            return {}
        
        on_demand_hourly = on_demand_pricing[instance_type]["pricing"]["ondemand_hourly"]
        discount_factor = ri_discounts.get(term, {}).get(payment_option, 0.25)
        
        ri_hourly = on_demand_hourly * (1 - discount_factor)
        
        return {
            "hourly_rate": ri_hourly,
            "monthly_rate": ri_hourly * 24 * 30,
            "annual_rate": ri_hourly * 24 * 365,
            "discount_percent": discount_factor * 100,
            "term": term,
            "payment_option": payment_option
        }
    
    def _parse_memory(self, memory_str: str) -> float:
        """Parse memory string to float value"""
        try:
            if isinstance(memory_str, (int, float)):
                return float(memory_str)
            
            # Handle strings like "8 GiB", "16.0 GiB"
            parts = str(memory_str).split()
            if len(parts) >= 1:
                return float(parts[0])
            return 0.0
        except (ValueError, AttributeError):
            return 0.0
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached data is still valid"""
        if not self.enable_caching or cache_key not in self.cache:
            return False
        
        return (time.time() - self.last_updated.get(cache_key, 0)) < self.CACHE_DURATION
    
    def _get_fallback_rds_pricing(self, region: str, engine: str) -> Dict:
        """Fallback pricing data when AWS API is unavailable"""
        
        fallback_data = {
            "postgres": {
                "db.t3.micro": {"type": "db.t3.micro", "vCPU": 2, "memory": 1, "pricing": {"ondemand_hourly": 0.0255, "ondemand_monthly": 18.4}, "instance_family": "t3"},
                "db.t3.small": {"type": "db.t3.small", "vCPU": 2, "memory": 2, "pricing": {"ondemand_hourly": 0.051, "ondemand_monthly": 36.7}, "instance_family": "t3"},
                "db.t3.medium": {"type": "db.t3.medium", "vCPU": 2, "memory": 4, "pricing": {"ondemand_hourly": 0.102, "ondemand_monthly": 73.4}, "instance_family": "t3"},
                "db.t3.large": {"type": "db.t3.large", "vCPU": 2, "memory": 8, "pricing": {"ondemand_hourly": 0.204, "ondemand_monthly": 146.9}, "instance_family": "t3"},
                "db.m5.large": {"type": "db.m5.large", "vCPU": 2, "memory": 8, "pricing": {"ondemand_hourly": 0.192, "ondemand_monthly": 138.2}, "instance_family": "m5"},
                "db.m5.xlarge": {"type": "db.m5.xlarge", "vCPU": 4, "memory": 16, "pricing": {"ondemand_hourly": 0.384, "ondemand_monthly": 276.5}, "instance_family": "m5"},
                "db.m5.2xlarge": {"type": "db.m5.2xlarge", "vCPU": 8, "memory": 32, "pricing": {"ondemand_hourly": 0.768, "ondemand_monthly": 553.0}, "instance_family": "m5"},
                "db.m5.4xlarge": {"type": "db.m5.4xlarge", "vCPU": 16, "memory": 64, "pricing": {"ondemand_hourly": 1.536, "ondemand_monthly": 1106.0}, "instance_family": "m5"},
                "db.r5.large": {"type": "db.r5.large", "vCPU": 2, "memory": 16, "pricing": {"ondemand_hourly": 0.24, "ondemand_monthly": 172.8}, "instance_family": "r5"},
                "db.r5.xlarge": {"type": "db.r5.xlarge", "vCPU": 4, "memory": 32, "pricing": {"ondemand_hourly": 0.48, "ondemand_monthly": 345.6}, "instance_family": "r5"},
                "db.r5.2xlarge": {"type": "db.r5.2xlarge", "vCPU": 8, "memory": 64, "pricing": {"ondemand_hourly": 0.96, "ondemand_monthly": 691.2}, "instance_family": "r5"}
            }
        }
        
        # Apply regional pricing multiplier
        regional_multipliers = {
            "us-east-1": 1.0, "us-west-1": 1.08, "us-west-2": 1.08,
            "eu-west-1": 1.15, "eu-central-1": 1.18,
            "ap-southeast-1": 1.20, "ap-northeast-1": 1.25
        }
        
        multiplier = regional_multipliers.get(region, 1.1)
        engine_data = fallback_data.get(engine, fallback_data["postgres"])
        
        # Apply regional adjustment
        adjusted_data = {}
        for instance_type, data in engine_data.items():
            adjusted_instance = data.copy()
            adjusted_instance["pricing"] = {
                "ondemand_hourly": data["pricing"]["ondemand_hourly"] * multiplier,
                "ondemand_monthly": data["pricing"]["ondemand_monthly"] * multiplier
            }
            adjusted_data[instance_type] = adjusted_instance
        
        self.logger.info(f"Using fallback pricing for {engine} in {region}")
        return adjusted_data
    
    def clear_cache(self):
        """Clear pricing cache"""
        self.cache.clear()
        self.last_updated.clear()
        self.logger.info("Pricing cache cleared")
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics"""
        return {
            "cached_items": len(self.cache),
            "cache_enabled": self.enable_caching,
            "cache_duration_seconds": self.CACHE_DURATION,
            "aws_available": self.available
        }