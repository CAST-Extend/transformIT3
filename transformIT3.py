import re
import os
import pdb
import time
import json
import glob
import shutil
import pickle
import zipfile
import requests
import linecache
import subprocess
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

from time import sleep
from pathlib import Path
from zipfile import ZipFile
from typing import List, Dict
from datetime import datetime
from itertools import groupby
from operator import itemgetter
from itertools import accumulate
from pandas import json_normalize
from collections import defaultdict
# from pydantic_settings import BaseSettings
# from pandas_profiling import ProfileReport
from requests.exceptions import RequestException

from misc_tools import my_datatable
from openai_tools import count_chatgpt_tokens

# Set display options to show all columns and rows
pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)

# Suppress scientific notation
pd.set_option('display.float_format', lambda x: '%.3f' % x)

count_token_mode = "not_tiktoken"
print(f"count_token_mode: {count_token_mode}")

openai_tool_file = os.path.join(os.getcwd(), "openai_tools.py")
if not os.path.exists(openai_tool_file):
    openai_tool_file = os.path.join(os.getcwd(), "..", "openai_tools.py")
exec(open(openai_tool_file).read())
print(f"openai_tool_file: {openai_tool_file}")

misc_tool_file = os.path.join(os.getcwd(), "misc_tools.py")
if not os.path.exists(misc_tool_file):
    misc_tool_file = os.path.join(os.getcwd(), "..", "misc_tools.py")
exec(open(misc_tool_file).read())

print(f"misc_tool_file: {misc_tool_file}")


# CONFIGURATION ####

# Fetch the target application from environment variables
target_application = os.getenv("TARGET_APPLICATION", "")
if not target_application:
    target_application = "webgoat_v8_2_2"

# Uncomment the lines below and modify as needed for different applications
# "shopizer_3_2_5"
# "shopizer_2_17_0"
# "shopizer_showcase_3_2_5"
# "webgoat_v8_2_2"
# "EMS"
# "DoNotShareMIB"
# "DoNotShareTKN"
# "DoNotShareCDC"
# "DoNotShareTRSP"

# Fetch the transformation source from environment variables
transformation_source = os.getenv("TARGET_TRANSFORMATION", "")
if not transformation_source:
    transformation_source = "cloud"

# Uncomment and modify the lines below for different sources
# "cloud"
# "green"
# "path"

# Determine the transformation target based on the transformation source
transformation_target = {
    "cloud": f"targeting AWS (list missing required information about the target situation in the dedicated field of your response)",
    # Uncomment and modify as needed for different transformations
    # "green": f"targeting Green (use specific credentials, '{target_application}' resource, 'us-west-2' region)",
}.get(transformation_source, "")

# Print or log the determined configuration
print(f"target_application: {target_application}")
print(f"transformation_source: {transformation_source}")
print(f"transformation_target: {transformation_target}")

# Directory for zip archives
zip_dir = Path.cwd() / "archives"

# Target source zip file path
target_source_zip = zip_dir / f"{target_application}.zip"

# Target source URL
target_source_url = (
    # Uncomment and modify the line below for different URLs
    # "https://github.com/shopizer-ecommerce/shopizer/archive/refs/tags/3.2.5.zip"
    # "https://github.com/shopizer-ecommerce/shopizer/archive/refs/tags/2.17.0.zip"
    # "https://github.com/marton-eifert/shopizer_showcase/archive/refs/heads/3.2.5-actual.zip"
    "https://github.com/WebGoat/WebGoat/archive/refs/tags/v8.2.2.zip"
    # "https://github.com/elastic/elasticsearch/archive/refs/tags/v8.12.2.zip"
    # None
)

# Source and transformed directories
source_dir = Path.cwd() / "input" / target_application
transformed_dir = Path.cwd() / "output" / target_application / transformation_source

# Print or log the paths for debugging purposes
print(f"zip_dir: {zip_dir}")
print(f"target_source_zip: {target_source_zip}")
print(f"target_source_url: {target_source_url}")
print(f"source_dir: {source_dir}")
print(f"transformed_dir: {transformed_dir}")

# Fetch environment variables
openai_model = os.getenv("OPENAI_MODEL", "")
openai_check_model = os.getenv("OPENAI_CHECK_MODEL", "")

mmc_azure_model = os.getenv("MMC_MODEL", "")
mmc_azure_check_model = os.getenv("MMC_CHECK_MODEL", "")
print(f"mmc_azure_check_model: {mmc_azure_check_model}")

azure_model = os.getenv("AZURE_MODEL", "")
azure_check_model = os.getenv("AZURE_CHECK_MODEL", "")
print(f"azure_model: {azure_model}")
print(f"azure_check_model: {azure_check_model}")

# Determine AI mode
if mmc_azure_model:
    ai_mode = "mmc"
elif azure_model:
    ai_mode = "azure"
elif target_application.startswith("DoNotShare"):
    ai_mode = "donotshare"
else:
    ai_mode = "openai"

# Set models based on AI mode
if ai_mode == "mmc":
    openai_model = mmc_azure_model
    openai_check_model = mmc_azure_check_model
elif ai_mode == "azure":
    openai_model = azure_model
    openai_check_model = azure_check_model
elif ai_mode == "donotshare":
    openai_model = "donotshare"

# Print or log the determined models and AI mode for debugging purposes
print(f"ai_mode: {ai_mode}")
print(f"openai_model: {openai_model}")
print(f"openai_check_model: {openai_check_model}")

print(f"ai_mode: {ai_mode}")
print(f"openai_model: {openai_model}")
print(f"openai_check_model: {openai_check_model}")

# Function to create directories if they don't exist
def create_directory_if_not_exists(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)

# Directory paths
carl_output_dir = os.path.join(os.getcwd(), "CARL_wrk", "output", target_application)
carl_cfg_dir = os.path.join(os.getcwd(), "CARL_wrk", "cfg")
carl_ext_dir = os.path.join(os.getcwd(), "CARL_wrk", "extensions")
hl_output_dir = os.path.join(os.getcwd(), "HL_wrk", "output", target_application)
gen_source_dir = os.path.join(os.getcwd(), "openAI_wrk", openai_model, "merged", target_application, transformation_source)
check_source_dir = os.path.join(os.getcwd(), "openAI_wrk", openai_check_model, "checked", target_application, transformation_source)
openai_run_trace_dir = os.path.join(os.getcwd(), "openAI_wrk", openai_model, "run", target_application, transformation_source)
openai_snippet_trace_dir = os.path.join(os.getcwd(), "openAI_wrk", openai_model, "snippets_wrk", target_application, transformation_source)
openai_check_trace_dir = os.path.join(os.getcwd(), "openAI_wrk", openai_check_model, "check_wrk", target_application, transformation_source)
openai_dependent_check_trace_dir = os.path.join(os.getcwd(), "openAI_wrk", openai_model, "dep_check_wrk", target_application, transformation_source)
print(f"carl_output_dir: {carl_output_dir}")
print(f"carl_cfg_dir: {carl_cfg_dir}")
print(f"carl_ext_dir: {carl_ext_dir}")
print(f"hl_output_dir: {hl_output_dir}")
print(f"gen_source_dir: {gen_source_dir}")
print(f"check_source_dir: {check_source_dir}")
print(f"openai_run_trace_dir: {openai_run_trace_dir}")
print(f"openai_snippet_trace_dir: {openai_snippet_trace_dir}")
print(f"openai_check_trace_dir: {openai_check_trace_dir}")
print(f"openai_dependent_check_trace_dir: {openai_dependent_check_trace_dir}")

# List of directory paths
directory_paths = [
    zip_dir, source_dir, transformed_dir,
    carl_output_dir, carl_cfg_dir, carl_ext_dir,
    hl_output_dir,
    gen_source_dir, check_source_dir,
    openai_run_trace_dir, openai_snippet_trace_dir, openai_check_trace_dir,
    openai_dependent_check_trace_dir
]

# Create directories if they don't exist and gather file information
directory_info = []
for dir_path in directory_paths:
    create_directory_if_not_exists(dir_path)
    dir_info = {
        'var': os.path.basename(dir_path),
        'dir_path': dir_path,
        'is_dir': os.path.isdir(dir_path),
        'size_bytes': os.path.getsize(dir_path) if os.path.isdir(dir_path) else None,
        'created_time': pd.to_datetime(os.path.getctime(dir_path), unit='s'),
        'modified_time': pd.to_datetime(os.path.getmtime(dir_path), unit='s')
    }
    directory_info.append(dir_info)

# Convert list of dictionaries to pandas DataFrame
directory_info_df = pd.DataFrame(directory_info)

# Define OpenAI model sizes using a dictionary
openai_model_sizes = {
    "gpt-4-turbo-preview": 128000,
    "gpt-4": 8192,
    "gpt-4-32k": 32768,
    "donotshare": 32768,
    "chatgpt432k": 32768,
    "mmc-tech-gpt-35-turbo": 8192,
    "mmc-tech-gpt-35-turbo-smart-latest": 8192,
    "gpt-3.5-turbo": 4096,
    "gpt-3.5-turbo-16k": 16384
}

# Create a DataFrame from the dictionary
openai_model_sizes_df = pd.DataFrame(list(openai_model_sizes.items()), columns=['model', 'model_size'])

# Print or use openai_model_sizes_df as needed
print(openai_model_sizes_df)

# Fetch environment variables for OpenAI API configuration
openai_api_url = os.getenv("OPENAI_API_URL", "")
openai_key = os.getenv("OPENAI_API_KEY", "")

mmc_azure_rsc = os.getenv("MMC_RSC", "")
mmc_azure_model = os.getenv("MMC_MODEL", "")
mmc_azure_version = os.getenv("MMC_VERSION", "")
print(f"mmc_azure_rsc: {mmc_azure_rsc}")
print(f"mmc_azure_model: {mmc_azure_model}")
print(f"mmc_azure_version: {mmc_azure_version}")

mmc_azure_api_url = f"{mmc_azure_rsc}/openai/v1/deployments/{mmc_azure_model}/chat/completions?api-version={mmc_azure_version}"
mmc_azure_key = os.getenv("MMC_API_KEY", "")
print(f"mmc_azure_api_url: {mmc_azure_api_url}")
print(f"mmc_azure_key: {mmc_azure_key}")

azure_rsc = os.getenv("AZURE_RSC", "")
azure_model = os.getenv("AZURE_MODEL", "")
azure_version = os.getenv("AZURE_VERSION", "")
print(f"azure_rsc: {azure_rsc}")
print(f"azure_model: {azure_model}")
print(f"azure_version: {azure_version}")

azure_api_url = f"{azure_rsc}/openai/deployments/{azure_model}/chat/completions?api-version={azure_version}"
azure_key = os.getenv("AZURE_API_KEY", "")
print(f"azure_api_url: {azure_api_url}")
print(f"azure_key: {azure_key}")

# Determine API URL and Key based on ai_mode
if ai_mode == "mmc":
    openai_key = mmc_azure_key
    openai_api_url = mmc_azure_api_url
elif ai_mode == "azure":
    openai_key = azure_key
    openai_api_url = azure_api_url
elif ai_mode == "donotshare":
    openai_key = "donotshare"

# Print or use openai_api_url and openai_key as needed
print(f"OpenAI API URL: {openai_api_url}")
print(f"OpenAI API Key: {openai_key}")

# Function to simulate the chat interaction (dummy function)
def ask_chatgpt_wo_functions(messages, model, url, api_key, debug):
    # Simulate API call and return response
    response = defaultdict(list)
    response['content'] = "Sure, I can help you with that!"
    return response

# Test messages and interaction
openai_test_messages = [{"role": "user", "content": "Hello! What can you do to help me?"}]
openai_test_prompt = [dict(message) for message in openai_test_messages]

# Call the function with the test data
openai_test_response = ask_chatgpt_wo_functions(messages=openai_test_prompt,
                                                model=openai_model,
                                                url=openai_api_url,
                                                api_key=openai_key,
                                                debug=False)

# Retrieve content from the response (dummy function assumed)
def get_chatgpt_content_from_response(response):
    return response['content']

# Print or use the response content
print(get_chatgpt_content_from_response(openai_test_response))

# Setting invocation delay
openai_invocation_delay = 10
print(f"openai_invocation_delay: {openai_invocation_delay}")

def hl_scan_it(source_dir, hl_output_dir, custom_mode=False, debug_mode=False):
    if debug_mode:
        breakpoint()

    # Adjust paths for Windows
    source_dir = os.path.abspath(source_dir).replace('\\', '/')
    hl_output_dir = os.path.abspath(hl_output_dir).replace('\\', '/')

    if custom_mode:
        java_conf_file = os.path.join(os.getcwd(), "HL_wrk", "conf", "Java_Conf.pm").replace('\\', '/')
        java_conf_file_image = "/opt/hlt/perl/Java_Conf.pm"
        
        ident_file = os.path.join(os.getcwd(), "HL_wrk", "conf", "Ident.pm").replace('\\', '/')
        ident_file_image = "/opt/hlt/perl/Ident.pm"
        
        java_count_custom_file = os.path.join(os.getcwd(), "HL_wrk", "conf", "Java", "CountCustom.pm").replace('\\', '/')
        java_count_custom_file_image = "/opt/hlt/perl/Java/CountCustom.pm"
        
        rule_desc_file = os.path.join(os.getcwd(), "HL_wrk", "conf", "GreenIT", "lib", "RulesDescription.csv").replace('\\', '/')
        rule_desc_file_image = "/opt/hlt/perl/GreenIT/lib/RulesDescription.csv"

        docker_args = [
            "docker", "run",
            "--rm" if not debug_mode else "",  "--user", "root", 
            f"-v \"{java_conf_file}:{java_conf_file_image}\"",
            f"-v \"{java_count_custom_file}:{java_count_custom_file_image}\"",
            f"-v \"{ident_file}:{ident_file_image}\"",
            f"-v \"{rule_desc_file}:{rule_desc_file_image}\"",
            f"-v \"{source_dir}:/sourceDir\"",
            f"-v \"{hl_output_dir}:/workingDir\"",
            "casthighlight/hl-agent-cli",
            "--skipUpload",
            "--dbgMatchPatternDetail",
            "--sourceDir", "/sourceDir",
            "--workingDir", "/workingDir"
        ]
    else:
        docker_args = [
            "docker", "run",
            "--rm" if not debug_mode else "",  "--user", "root", 
            f"-v \"{source_dir}:/sourceDir\"",
            f"-v \"{hl_output_dir}:/workingDir\"",
            "casthighlight/hl-agent-cli",
            "--skipUpload",
            "--dbgMatchPatternDetail",
            "--sourceDir", "/sourceDir",
            "--workingDir", "/workingDir"
        ]

    # Filter out empty strings
    docker_args = [arg for arg in docker_args if arg]

    print("Running Docker command:")
    print(" ".join(docker_args))

    hl_docker_run_trace = subprocess.run(
        " ".join(docker_args),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        shell=True  # Add shell=True to ensure proper handling of the command
    )

    print("Stdout:")
    print(hl_docker_run_trace.stdout)
    print("Stderr:")
    print(hl_docker_run_trace.stderr)
    
# Configuration
reset_input_sources = True
max_line_span = 30
min_line_span = 20
debug_flow = False
debug_loop = False
print(f"max_line_span: {max_line_span}")
print(f"min_line_span: {min_line_span}")
print(f"debug_flow: {debug_flow}")
print(f"debug_loop: {debug_loop}")

run_whole_file_check = False

def download_file(url, destfile):
    response = requests.get(url)
    with open(destfile, 'wb') as f:
        f.write(response.content)

def unzip_file(zipfile_path, extract_to):
    with zipfile.ZipFile(zipfile_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

def carl_scan_it(target_application, source_dir, carl_output_dir, carl_cfg_dir, carl_ext_dir, debug_mode=False):
    if debug_mode:
        import pdb
        pdb.set_trace()

    # Convert paths to absolute paths and replace backslashes with forward slashes
    source_dir = os.path.abspath(source_dir).replace('\\', '/')
    carl_output_dir = os.path.abspath(carl_output_dir).replace('\\', '/')
    carl_cfg_dir = os.path.abspath(carl_cfg_dir).replace('\\', '/')
    carl_ext_dir = os.path.abspath(carl_ext_dir).replace('\\', '/')
    
    carl_cfg_file = os.path.join(carl_cfg_dir, "configFile.xml").replace('\\', '/')
    carl_run_file = os.path.join(carl_cfg_dir, "run.sh").replace('\\', '/')
    carl_image_ext_path = '/home/carl/add_extensions'
    carl_image = "castbuild/carl:release"  # Ensure this is in lowercase

    # Check if the directory exists
    if os.path.exists(carl_output_dir):
        # If it exists, remove it
        shutil.rmtree(carl_output_dir)

    # Constructing Docker command arguments
    docker_args = [
        "docker", "run",
        "--rm" if not debug_mode else "",  # Add --rm only if not in debug_mode
        "-e", f"APP={target_application.lower().replace(' ', '_')}",  # Ensure app name is lowercase and has no spaces
        "-v", f'"{source_dir}:/home/carl/sources"',
        "-v", f'"{carl_output_dir}:/home/carl/output"',
        "-v", f'"{carl_ext_dir}:{carl_image_ext_path}"',
        "-v", f'"{carl_cfg_file}:/home/carl/configFile.xml"',
        "-v", f'"{carl_run_file}:/home/carl/run.sh"',
        "--entrypoint", "run.sh",
        carl_image.lower()  # Ensure the image name is in lowercase
    ]

    # Remove empty strings in docker_args
    docker_args = [arg for arg in docker_args if arg]

    # Joining docker_args into a single string command
    docker_command = " ".join(docker_args)
    print("Running CARL Docker command:", docker_command)

    # Running the Docker command using subprocess
    try:
        carl_docker_run_trace = subprocess.run(
            docker_command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True  # Raise exception on non-zero return code
        )
        print(carl_docker_run_trace.stdout.decode('utf-8'))
    except subprocess.CalledProcessError as e:
        print(f"Error running CARL Docker command: {e}")
        print(e.stderr.decode('utf-8'))

if len(glob.glob(os.path.join(source_dir, '*'))) == 0 or reset_input_sources:
    if not os.path.exists(target_source_zip):
        download_file(target_source_url, target_source_zip)
    unzip_file(target_source_zip, source_dir)
    
    print(datetime.now())
    hl_scan_it(source_dir, hl_output_dir, debug_mode=debug_flow)
    print(datetime.now())
    
    shutil.rmtree(carl_output_dir, ignore_errors=True)
    if os.path.exists(carl_output_dir):
        breakpoint()
    carl_scan_it(target_application, source_dir, carl_output_dir, carl_cfg_dir, carl_ext_dir, debug_mode=debug_flow)
    print(datetime.now())
    
    shutil.rmtree(gen_source_dir, ignore_errors=True)
    if os.path.exists(gen_source_dir):
        breakpoint()

def hl_document_cloud_patterns(carl_ext_dir, debug_mode=False):
    if debug_mode:
        import pdb
        pdb.set_trace()

    # Construct the path to the JSON document
    hl_cloud_pattern_doc_file = os.path.join(
        carl_ext_dir,
        "com.castsoftware.highlight2mri",
        "documentation",
        "cloudready-patterns.json"
    )

    # Check if the JSON file exists
    if os.path.exists(hl_cloud_pattern_doc_file):
        # Read and parse the JSON file
        with open(hl_cloud_pattern_doc_file, 'r', encoding='utf-8') as f:
            hl_pattern_doc = json.load(f)

        # Function to process nested JSON structure and flatten into DataFrame
        def process_json(json_obj):
            rows = []
            if isinstance(json_obj, dict):
                json_obj = [json_obj]
            for entry in json_obj:
                base_info = {
                    'req_id': entry.get('key'),
                    'req_title': entry.get('title'),
                    'req_desc': entry.get('platformMigration', {}).get('description'),
                    'req_ref': entry.get('platformMigration', {}).get('reference'),
                    'req_nat': entry.get('contribution', {}).get('name')
                }
                for scope in entry.get('scope', []):
                    row = base_info.copy()
                    row.update({
                        'technology': scope.get('technology', {}).get('name'),
                        'technology_code': scope.get('technology', {}).get('code'),
                        'req_pattern': scope.get('searchedCodePatterns')
                    })
                    rows.append(row)
            return pd.DataFrame(rows)

        # Process the JSON structure into a DataFrame
        hl_pattern_req_instructions = process_json(hl_pattern_doc)

        # Select relevant columns (if they exist in the DataFrame)
        required_columns = [
            'req_id', 'req_title', 'technology', 'technology_code',
            'req_pattern', 'req_desc', 'req_ref', 'req_nat'
        ]

        # Filter and extract unique blocker and booster IDs
        if not hl_pattern_req_instructions.empty:
            hl_pattern_req_instructions = hl_pattern_req_instructions[required_columns]

            hl_cloud_blocker_ids = hl_pattern_req_instructions.loc[
                hl_pattern_req_instructions['req_nat'] == 'Blocker', 'req_id'
            ].unique()

            hl_cloud_booster_ids = hl_pattern_req_instructions.loc[
                hl_pattern_req_instructions['req_nat'] == 'Booster', 'req_id'
            ].unique()

            # Return as a dictionary
            return {
                'hl_pattern_req_instructions': hl_pattern_req_instructions,
                'hl_cloud_blocker_ids': hl_cloud_blocker_ids,
                'hl_cloud_booster_ids': hl_cloud_booster_ids
            }
        else:
            print("No data available after processing JSON.")
            return None
    else:
        print(f"File not found: {hl_cloud_pattern_doc_file}")
        return None

def hl_process_cloud_log_files(hl_output_dir, hl_cloud_blocker_ids, debug_mode=False):
    if debug_mode:
        import pdb
        pdb.set_trace()

    hl_log_dir = os.path.join(hl_output_dir, "HLTemporary", "analysis")

    if os.path.exists(hl_log_dir):
        # List all cloud log files matching the pattern
        cloud_log_files = [os.path.join(hl_log_dir, f) for f in os.listdir(hl_log_dir) if re.match(r"^cloudDetail_[\w]*\.csv$", f)]

        def read_lines(file_path):
            with open(file_path, 'r') as file:
                return file.readlines()

        cloud_log_content = pd.concat(
            [
                pd.DataFrame({
                    'scan_path': [os.path.basename(file)] * len(read_lines(file)),
                    'line': range(1, len(read_lines(file)) + 1),
                    'content': read_lines(file)
                })
                for file in cloud_log_files
            ],
            ignore_index=True
        )
        cloud_log_content['content'] = cloud_log_content['content'].apply(lambda x: x.strip())

        # Step 1: Separate scan_path using regex patterns
        cloud_pattern_occurrence_locations = cloud_log_content.copy()
        cloud_pattern_occurrence_locations['technology'] = cloud_pattern_occurrence_locations['scan_path'].str.extract(r'cloudDetail_(\w*)\.csv')[0]        

        # print(cloud_pattern_occurrence_locations)

        # Step 2: Mark and filter bookmarks
        cloud_pattern_occurrence_locations['is_bookmark'] = (
            cloud_pattern_occurrence_locations['content'] == "Alert;Path;LineNumberPos;Url_doc;Kind"
        ).astype(int)
        cloud_pattern_occurrence_locations['is_bookmark'] = (
            cloud_pattern_occurrence_locations.groupby('scan_path')['is_bookmark']
            .cumsum()
        )
        cloud_pattern_occurrence_locations = cloud_pattern_occurrence_locations[
            (cloud_pattern_occurrence_locations['is_bookmark'] > 0) &
            (cloud_pattern_occurrence_locations['content'] != "Alert;Path;LineNumberPos;Url_doc;Kind")
        ]

        # print(cloud_pattern_occurrence_locations)

        # Step 3: Separate and process content columns
        cloud_pattern_occurrence_locations[['req_name', 'source_path', 'LineNumberPos', 'url_doc', 'req_nat']] = (
            cloud_pattern_occurrence_locations['content']
            .str.split(';', expand=True)
            .apply(lambda x: x.str.strip())
        )

        # print(cloud_pattern_occurrence_locations)

        # Step 4: Additional processing on req_name
        cloud_pattern_occurrence_locations['req_name'] = (
            cloud_pattern_occurrence_locations['req_name']
            .str.replace(r'^[[:punct:]]*', '')
            .str.replace(r'[[:punct:]]*$', '')
        )

        # print(cloud_pattern_occurrence_locations)

        # Step 5: Separate url_doc and LineNumberPos columns
        cloud_pattern_occurrence_locations[['url_doc', 'req_id']] = (
            cloud_pattern_occurrence_locations['url_doc']
            .str.split('#', expand=True)
        )
        # print(cloud_pattern_occurrence_locations)

        # cloud_pattern_occurrence_locations[['LineNumberPos']] = (
        #     cloud_pattern_occurrence_locations['LineNumberPos']
        #     .str.split('|', expand=True)
        # )
        # print(cloud_pattern_occurrence_locations)

        cloud_pattern_occurrence_locations[['line_number', 'col_start', 'col_end']] = (
            cloud_pattern_occurrence_locations['LineNumberPos']
            .str.extract(r'(\d+)\[(\d+),(\d+)\]', expand=True)  # Extract numeric parts using regex

        )
        # print(cloud_pattern_occurrence_locations)

        # Step 6: Filter based on req_id in hl_cloud_blocker_ids
        cloud_pattern_occurrence_locations = cloud_pattern_occurrence_locations[
            cloud_pattern_occurrence_locations['req_id'].isin(hl_cloud_blocker_ids)
        ]

        # Display the resulting DataFrame
        # print(cloud_pattern_occurrence_locations)

        # Retrieve unique source paths and process each file
        # Function to handle file paths
        def adjust_file_path(file, source_dir):
            flagged_file = file
            if not os.path.exists(flagged_file):
                flagged_file = os.path.join(source_dir, file.replace("^/sourceDir/", ""))
            if not os.path.exists(flagged_file):
                # Use browser() for debugging if necessary
                pass  # Replace with actual debugging steps if needed
            return flagged_file

        # Function to process each file
        def process_file(file):
            flagged_file = adjust_file_path(file, source_dir)
            
            # Read cloud pattern occurrences for the current file
            file_cloud_findings = cloud_pattern_occurrence_locations.loc[
                cloud_pattern_occurrence_locations['source_path'] == file,
                ['technology', 'line_number', 'req_id']
            ].drop_duplicates().groupby('line_number').agg(lambda x: '||'.join(x.unique())).reset_index()
            
            # Convert line_number to int64 to match merged_data
            file_cloud_findings['line_number'] = file_cloud_findings['line_number'].astype(np.int64)
            
            file_cloud_findings['flagged'] = 1
            
            # Read flagged content from the file
            with open(flagged_file, 'r') as f:
                flagged_content = f.readlines()
            
            # Join flagged content with cloud findings
            merged_data = pd.DataFrame({'line_number': np.arange(len(flagged_content)), 'code': flagged_content})
            merged_data = pd.merge(merged_data, file_cloud_findings, on='line_number', how='left')
            merged_data['flagged'].fillna(0, inplace=True)
            merged_data['technology'].fillna('', inplace=True)
            merged_data['req_id'].fillna('', inplace=True)
            
            # Calculate near flagged lines
            merged_data['near_flagged'] = merged_data['flagged'].rolling(min_line_span, center=True, min_periods=1).sum()
            merged_data['group_line'] = merged_data['near_flagged'] > 0
            merged_data['group_idx'] = (merged_data['group_line'] != merged_data['group_line'].shift()).astype(int).cumsum()
            
            # Process groups with flagged content
            processed_data = merged_data.loc[merged_data['flagged'] > 0].groupby('group_idx').agg({
                'line_number': ['min', 'max'],
                'technology': lambda x: ''.join(x.unique()),
                'code': lambda x: '\n'.join(x),
                'req_id': lambda x: '||'.join(x.unique())
            }).reset_index()
            processed_data.columns = ['group_idx', 'line_start', 'line_end', 'technology', 'code', 'req_id_list']
            processed_data['technology'] = processed_data['technology'].str.replace(r'^\|\||\|\|$', '')
            processed_data['req_id_list'] = processed_data['req_id_list'].str.replace(r'^\|\||\|\|$', '')
            processed_data['abs_source_path'] = flagged_file
            processed_data['line_span'] = processed_data['line_end'] - processed_data['line_start']
            processed_data['source_path'] = file
            
            return processed_data

        # Apply function to each unique source_path
        unique_files = cloud_pattern_occurrence_locations['source_path'].unique()
        cloud_pattern_occurrence_snippets = pd.concat([process_file(file) for file in unique_files], ignore_index=True)

        print(f"cloud_log_files: {cloud_log_files}")
        print(f"cloud_log_content: {cloud_log_content}")
        print(f"cloud_pattern_occurrence_locations: {cloud_pattern_occurrence_locations}")
        print(f"cloud_pattern_occurrence_snippets: {cloud_pattern_occurrence_snippets}")

        return {
            'cloud_log_files': cloud_log_files,
            'cloud_log_content': cloud_log_content,
            'cloud_pattern_occurrence_locations': cloud_pattern_occurrence_locations,
            'cloud_pattern_occurrence_snippets': cloud_pattern_occurrence_snippets
        }
    else:
        print(f"Directory not found: {hl_log_dir}")
        return None    


def hl_process_green_log_files(hl_output_dir, debug_mode=False):
    if debug_mode:
        import pdb; pdb.set_trace()  # Debug mode similar to R's browser()
    
    hl_log_dir = os.path.join(hl_output_dir, "HLTemporary", "analysis")
    
    if os.path.exists(hl_log_dir):
        green_log_files = [f for f in os.listdir(hl_log_dir) if re.match(r"^greenDetail_[[:alnum:]]*\.csv", f)]
        green_log_files = [os.path.join(hl_log_dir, f) for f in green_log_files]
        
        green_log_content = pd.concat([
            pd.read_csv(f, header=None, names=["content"]).assign(scan_path=f) for f in green_log_files
        ], ignore_index=True)
        
        green_url_docs = (
            green_log_content["scan_path"]
            .str.extract(r"^.*greenDetail_(?P<technology>[[:alnum:]]*)\.csv$")
            .join(green_log_content["content"].str.split(";", expand=True))
            .rename(columns={0: "req_name", 1: "source_path", 2: "LineNumberPos", 3: "url_doc"})
            .groupby("scan_path")
            .filter(lambda x: (x["content"] == "Alert;Path;LineNumberPos;Url_doc").any())
            .assign(is_bookmark=lambda x: (x["content"] == "Alert;Path;LineNumberPos;Url_doc").cumsum())
            .query("is_bookmark > 0 and content != 'Alert;Path;LineNumberPos;Url_doc'")
            .drop("is_bookmark", axis=1)
            .assign(req_name=lambda x: x["req_name"].str.replace(r"^[[:punct:]]*", "").str.replace(r"[[:punct:]]*$", ""))
            .drop_duplicates(subset=["req_name", "url_doc"])
            [["req_name", "url_doc"]]
        )
        
        green_pattern_occurrence_locations = (
            green_log_content["scan_path"]
            .str.extract(r"^.*greenDetail_(?P<technology>[[:alnum:]]*)\.csv$")
            .join(green_log_content["content"].str.split(";", expand=True))
            .rename(columns={0: "req_name", 1: "source_path", 2: "LineNumberPos", 3: "url_doc"})
            .groupby("scan_path")
            .filter(lambda x: (x["content"] == "Alert;Path;LineNumberPos;Url_doc").any())
            .assign(is_bookmark=lambda x: (x["content"] == "Alert;Path;LineNumberPos;Url_doc").cumsum())
            .query("is_bookmark > 0 and content != 'Alert;Path;LineNumberPos;Url_doc'")
            .drop("is_bookmark", axis=1)
            .assign(req_name=lambda x: x["req_name"].str.replace(r"^[[:punct:]]*", "").str.replace(r"[[:punct:]]*$", ""))
            .assign(req_id=lambda x: x["url_doc"].str.split("#", expand=True)[1])
            .assign(LineNumberPos=lambda x: x["LineNumberPos"].str.split("|"))
            .explode("LineNumberPos")
            .assign(line_number=lambda x: x["LineNumberPos"].str.split(r"[[:punct:]]", expand=True)[0].astype(int),
                    col_start=lambda x: x["LineNumberPos"].str.split(r"[[:punct:]]", expand=True)[1].astype(int),
                    col_end=lambda x: x["LineNumberPos"].str.split(r"[[:punct:]]", expand=True)[2].astype(int))
            .drop("LineNumberPos", axis=1)
        )
        
        green_pattern_occurrence_snippets = []
        for file in green_pattern_occurrence_locations["source_path"].unique():
            flagged_file = file
            if not os.path.exists(flagged_file):
                flagged_file = os.path.join(source_dir, file.removeprefix("/sourceDir/"))
            if not os.path.exists(flagged_file):
                if debug_mode:
                    pdb.set_trace()  # Debug if file still not found
                
            file_green_findings = (
                green_pattern_occurrence_locations
                .loc[green_pattern_occurrence_locations["source_path"] == file, ["technology", "line_number", "req_id"]]
                .drop_duplicates()
                .groupby("line_number")
                .agg(lambda x: "||".join(x.unique()))
                .assign(flagged=1)
            )
            
            flagged_content = pd.read_csv(flagged_file, header=None, names=["code"]).assign(line_number=lambda x: x.index + 1)
            
            snippets = (
                flagged_content
                .merge(file_green_findings, on="line_number", how="left")
                .assign(flagged=lambda x: x["flagged"].fillna(0).astype(int),
                        technology=lambda x: x["technology"].fillna(""),
                        req_id=lambda x: x["req_id"].fillna(""))
                .assign(near_flagged=lambda x: x["flagged"].rolling(min_line_span, center=True).sum().fillna(0).astype(int))
                .assign(group_line=lambda x: x["near_flagged"] > 0)
                .assign(group_idx=lambda x: x["group_line"].ne(x["group_line"].shift(fill_value=False)).cumsum())
                .groupby("group_idx")
                .filter(lambda x: x["flagged"].any())
                .groupby("group_idx")
                .agg(line_start=("line_number", "min"),
                     line_end=("line_number", "max"),
                     technology=("technology", lambda x: "||".join(x.unique())),
                     code=("code", lambda x: "\n".join(x)),
                     req_id_list=("req_id", lambda x: "||".join(x.unique())))
                .assign(abs_source_path=flagged_file,
                        line_span=lambda x: x["line_end"] - x["line_start"])
                .reset_index(drop=True)
            )
            
            green_pattern_occurrence_snippets.append(snippets)
        
        green_pattern_occurrence_snippets = pd.concat(green_pattern_occurrence_snippets, ignore_index=True)
        
        return {
            "green_log_files": green_log_files,
            "green_log_content": green_log_content,
            "green_pattern_occurrence_locations": green_pattern_occurrence_locations,
            "green_pattern_occurrence_snippets": green_pattern_occurrence_snippets,
            "green_url_docs": green_url_docs
        }
    
    return None  # Return None if hl_log_dir does not exist or no files found

# Simulating hl_documentation
hl_documentation = hl_document_cloud_patterns(carl_ext_dir, debug_mode=False)

hl_pattern_req_instructions = hl_documentation["hl_pattern_req_instructions"]
hl_cloud_blocker_ids = hl_documentation["hl_cloud_blocker_ids"]
hl_cloud_booster_ids = hl_documentation["hl_cloud_booster_ids"]

print(f"hl_pattern_req_instructions: {hl_pattern_req_instructions}")
print(f"hl_cloud_blocker_ids: {hl_cloud_blocker_ids}")
print(f"hl_cloud_booster_ids: {hl_cloud_booster_ids}")

def carl_document_patterns(carl_output_dir, debug_mode=False):
    if debug_mode:
        import pdb; pdb.set_trace()

    carl_pattern_doc_file = os.path.join(carl_output_dir, "RuleSummary.json")

    carl_pattern_url = "https://technologies.castsoftware.com/rest/AIP/quality-rules/"

    if os.path.exists(carl_pattern_doc_file):
        with open(carl_pattern_doc_file, 'r') as f:
            carl_pattern_doc = json.load(f)
        
        carl_pattern_doc = pd.json_normalize(carl_pattern_doc, record_path=['qualityRules', 'rules'], 
                                             meta=['technology'])
        
        carl_pattern_doc = carl_pattern_doc.rename(columns={"violationId": "req_id", "name": "req_title"})
        carl_pattern_doc['req_id'] = carl_pattern_doc['req_id'].astype(str)

        violation_ids = carl_pattern_doc[~carl_pattern_doc['req_id'].isna()]['req_id'].unique()

        def fetch_rule_details(violation_id):
            try:
                print(violation_id)
                response = requests.get(f"{carl_pattern_url}{violation_id}")
                if response.status_code == 200 and response.text != "Not Found":
                    return json.loads(f"[{response.text}]")
            except Exception as e:
                print(f"Error fetching details for violation ID {violation_id}: {e}")
            return None

        more_docs = []
        for violation_id in violation_ids:
            rule_details = fetch_rule_details(violation_id)
            if rule_details:
                for rule in rule_details:
                    rule['aipId'] = violation_id
                    more_docs.append(rule)

        if more_docs:
            carl_pattern_more_doc = pd.json_normalize(more_docs)
            carl_pattern_more_doc = carl_pattern_more_doc[['id', 'name', 'description', 'output', 'rationale', 'reference']]
            carl_pattern_more_doc = carl_pattern_more_doc.rename(columns={"id": "req_id"})
            carl_pattern_more_doc['req_id'] = carl_pattern_more_doc['req_id'].astype(str)

            carl_pattern_req_instructions = carl_pattern_doc.assign(
                req_nat="Blocker"
            ).merge(
                carl_pattern_more_doc[['req_id', 'reference', 'rationale', 'description']],
                on='req_id',
                how='left',
                suffixes=('', '_more')
            ).rename(columns={
                'reference': 'req_ref',
                'rationale': 'req_desc',
                'description': 'req_pattern'
            })

            carl_blocker_ids = carl_pattern_req_instructions[carl_pattern_req_instructions['req_nat'] == 'Blocker']['req_id'].unique()
            carl_booster_ids = carl_pattern_req_instructions[carl_pattern_req_instructions['req_nat'] == 'Booster']['req_id'].unique()

            return {
                'carl_pattern_req_instructions': carl_pattern_req_instructions,
                'carl_blocker_ids': carl_blocker_ids,
                'carl_booster_ids': carl_booster_ids
            }

def carl_process_metamodel(carl_output_dir, debug_mode=False):
    if debug_mode:
        import pdb; pdb.set_trace()
    
    carl_mm_file = os.path.join(carl_output_dir, "config", "MetaModelStore.json")
    
    def read_json_file(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    
    carl_data = read_json_file(carl_mm_file)
    # print(carl_data)
    
    carl_types = pd.json_normalize(carl_data['alltypes'])
    carl_cats = pd.json_normalize(carl_data['allcats'])
    carl_props = pd.json_normalize(carl_data['allprops'])
    # # Remove duplicate rows
    # carl_types = carl_types.drop_duplicates()
    # print(carl_types)
    
    # def extract_data(data, key):
    #     if key in data:
    #         return pd.json_normalize(data[key])
    #     else:
    #         return pd.DataFrame()
    
    # carl_types = extract_data(flattened_data, 'alltypes').drop_duplicates().reset_index(drop=True)
    # carl_cats = extract_data(flattened_data, 'allcats').drop_duplicates().reset_index(drop=True)
    # carl_props = extract_data(flattened_data, 'allprops').drop_duplicates().reset_index(drop=True)
    
    return {
        'carl_types': carl_types,
        'carl_cats': carl_cats,
        'carl_props': carl_props
    }

def carl_process_repository_results(carl_output_dir, debug_mode=False):
    if debug_mode:
        import pdb; pdb.set_trace()
    
    carl_object_files = glob.glob(os.path.join(carl_output_dir, "repository", "**", "ObjectDescriptors_*.json"), recursive=True)

    def load_json(json_path):
        with open(json_path) as f:
            return json.load(f)

    def process_parents(file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Extract objects and parents information
        objects = data.get('objects', [])
        parents = []
        
        for obj in objects:
            obj_data = {
                'scan_id': obj.get('AipId'),
                'fullname': obj.get('fullname'),
                'name': obj.get('name'),
                'techno': obj.get('techno'),
                'source_path': obj.get('path'),
                'type': obj.get('type'),
                'type_str': obj.get('typeStr')
            }
            
            if 'parent' in obj:
                for parent in obj['parent']:
                    parent_data = {
                        'scan_id': obj.get('AipId'),
                        'fullname': obj.get('fullname'),
                        'name': obj.get('name'),
                        'techno': obj.get('techno'),
                        'source_path': obj.get('path'),
                        'type': obj.get('type'),
                        'type_str': obj.get('typeStr'),
                        'parent': parent
                    }
                    parents.append(parent_data)

            df = pd.DataFrame(parents)
            df["scan_path"] = file_path

        return df

    def process_args(file_path):
        # print(file_path)
        
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        objects = data.get('objects', [])
        records = []
        
        for obj in objects:
            base_record = {
                'scan_id': obj.get('AipId'),
                'fullname': obj.get('fullname'),
                'name': obj.get('name'),
                'techno': obj.get('techno'),
                'source_path': obj.get('path'),
                'type': obj.get('type'),
                'type_str': obj.get('typeStr')
            }
            
            parameters = obj.get('parameters', [])
            for param in parameters:
                record = base_record.copy()
                record.update({
                    'arg_position': param.get('position'),
                    'arg_name': param.get('name'),
                    'arg_type': param.get('type')
                })
                records.append(record)
        
        df = pd.DataFrame(records)
        if 'position' in df.columns:
            df = df[['scan_id', 'fullname', 'name', 'techno', 'source_path', 'type', 'type_str', 'arg_position', 'arg_name', 'arg_type']].drop_duplicates()
        df['scan_path'] = file_path

        return df

    def process_projects(file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)        
        result = []
        for obj in data['objects']:
            for proj in obj['project']:
                src_path = obj.get("path", "")
                result.append({
                    'scan_id': obj['AipId'],
                    'fullname': obj['fullname'],
                    'name': obj['name'],
                    'techno': obj['techno'],
                    'source_path': src_path,
                    'type': obj['type'],
                    'type_str': obj['typeStr'],
                    'project': proj['id'],
                    'external': proj['external']
                })
        df = pd.DataFrame(result).drop_duplicates()
        df['scan_path'] = file_path
        return df

    def process_positions(json_path):
        # print(json_path)
        data = load_json(json_path)
        # Flatten objects
        object_rows = []
        for obj in data['objects']:
            for bookmark in obj.get('bookmarks', []):
                for idx, pos in enumerate(bookmark.get('pos', [])):
                    object_rows.append({
                        'scan_id': obj.get('AipId'),
                        'fullname': obj.get('fullname'),
                        'name': obj.get('name'),
                        'techno': obj.get('techno'),
                        'source_path': obj.get('path'),
                        'type': obj.get('type'),
                        'type_str': obj.get('typeStr'),
                        'pos_source_path': bookmark.get('filePath'),
                        'file_id': bookmark.get('fileId'),
                        'pos_idx': idx + 1,
                        'pos': pos
                    })        
        df = pd.DataFrame(object_rows)
        df = df[df['pos'] > 0]
        df_wide = df.pivot_table(
            index=['scan_id', 'fullname', 'name', 'techno', 'source_path', 'type', 'type_str', 'pos_source_path', 'file_id'],
            columns='pos_idx',
            values='pos',
            aggfunc=lambda x: '#'.join(map(str, x))
        ).reset_index()        
        df_wide.columns = [f"v{col}" if isinstance(col, int) else col for col in df_wide.columns]
        df_wide['scan_path'] = json_path        
        return df_wide

    def process_callees(file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)       
        records = []
        for obj in data.get('objects', []):
            scan_id = obj.get('AipId')
            fullname = obj.get('fullname')
            name = obj.get('name')
            techno = obj.get('techno')
            source_path = obj.get('path')
            obj_type = obj.get('type')
            type_str = obj.get('typeStr')            
            if 'callees' in obj:
                for callee in obj['callees']:
                    record = {
                        'scan_path': file_path,
                        'scan_id': scan_id,
                        'fullname': fullname,
                        'name': name,
                        'techno': techno,
                        'source_path': source_path,
                        'type': obj_type,
                        'type_str': type_str,
                        'link_type': callee.get('linkType'),
                        'link_type_str': callee.get('linkTypeStr'),
                        'callee_project': callee.get('project'),
                        'callee': callee.get('AipId')
                    }
                    if 'pos' in callee:
                        for i, pos in enumerate(callee['pos']):
                            record[f'v{i+1}'] = pos                    
                    records.append(record)        
        return pd.DataFrame(records)

    carl_object_parents = pd.concat([process_parents(file) for file in carl_object_files])
    carl_object_args = pd.concat([process_args(file) for file in carl_object_files])
    carl_object_projects = pd.concat([process_projects(file) for file in carl_object_files])
    carl_object_positions = pd.concat([process_positions(file) for file in carl_object_files])
    carl_object_callees = pd.concat([process_callees(file) for file in carl_object_files])

    carl_object_parents['scan_id'] = carl_object_parents['scan_id'].astype(int)
    carl_object_parents['type'] = carl_object_parents['type'].astype(int)
    carl_object_parents['parent'] = carl_object_parents['parent'].astype(int)

    carl_object_args['scan_id'] = carl_object_args['scan_id'].astype(int)
    carl_object_args['type'] = carl_object_args['type'].astype(int)
    carl_object_args['arg_position'] = carl_object_args['arg_position'].astype(int)

    return {
        'carl_object_projects': carl_object_projects,
        'carl_object_args': carl_object_args,
        'carl_object_positions': carl_object_positions,
        'carl_object_callees': carl_object_callees,
        'carl_object_parents': carl_object_parents
    }

def carl_process_file_results(carl_output_dir, debug_mode=False):
    if debug_mode:
        import pdb; pdb.set_trace()  # Debug mode similar to R's browser()

    carl_file_files = []
    result_by_file_dir = os.path.join(carl_output_dir, "resultByFile")
    for root, _, files in os.walk(result_by_file_dir):
        for file in files:
            if file.endswith(".json"):
                carl_file_files.append(os.path.join(root, file))
    
    carl_file_violations_list = []

    for file_path in carl_file_files:
        print(file_path)
        with open(file_path, 'r') as file:
            data = json.load(file)
        
        violations = []
        source_path = data.get('file')
        for violation_list in data.get('ViolationList', []):
            for violation in violation_list.get('Violations', []):
                violation_record = {
                    'source_path': source_path,
                    'insight': violation.get('ViolationName'),
                    'insight_id': violation.get('ViolationId'),
                    'nb_occurrences': violation.get('ViolationsCount'),
                    'occurrence_id': violation.get('ID')
                }
                bookmarks = violation.get('bookmarks', [])
                for bookmark in bookmarks:
                    bookmark_record = violation_record.copy()
                    bookmark_record.update({
                        'start_line': bookmark.get('lineStart'),
                        'end_line': bookmark.get('lineEnd')
                    })
                    violations.append(bookmark_record)
                if not bookmarks:
                    violations.append(violation_record)
        
        carl_file_violations_list.extend(violations)
    
    carl_file_violations = pd.DataFrame(carl_file_violations_list)
    
    return {'carl_file_violations': carl_file_violations}

def carl_process_path_results(carl_output_dir, debug_mode=False):
    if debug_mode:
        import pdb; pdb.set_trace()  # Debug mode similar to R's browser()

    carl_path_files = []
    result_with_path_dir = os.path.join(carl_output_dir, "resultWithPath")
    for root, _, files in os.walk(result_with_path_dir):
        for file in files:
            if file.endswith(".json"):
                carl_path_files.append(os.path.join(root, file))
    
    path_pattern_occurrence_locations_list = []

    for file_path in carl_path_files:
        print(file_path)
        with open(file_path, 'r') as file:
            data = json.load(file)
        
        for bookmark in data.get('bookmarks', []):
            bookmark_record = {
                'req_name': bookmark.get('ViolationName'),
                'req_id': str(bookmark.get('ViolationId')),
                'occurrence_id': bookmark.get('ID'),
                'step': bookmark.get('step'),
                'source_path': bookmark.get('path'),
                'line_number': bookmark.get('lineStart'),
                'end_line_number': bookmark.get('lineEnd'),
                'col_start': bookmark.get('colStart'),
                'col_end': bookmark.get('colEnd')
            }
            path_pattern_occurrence_locations_list.append(bookmark_record)
    
    path_pattern_occurrence_locations = pd.DataFrame(path_pattern_occurrence_locations_list)
    
    return {'path_pattern_occurrence_locations': path_pattern_occurrence_locations}

def clean_source_path(df):
    if 'source_path' in df.columns:
        df['source_path'] = df['source_path'].str.replace(r"/home/carl/(sources|output|sources_wrk|output_wrk)/", "", regex=True)
    return df

def get_object_range(carl_object_positions, object_path_list, debug_mode):
    if debug_mode:
        import pdb; pdb.set_trace()

    # result = []
    # for source_path in set(object_path_list):
    #     # code_lines = pd.read_csv(os.path.join(source_dir, source_path), header=None, names=["code"])
    #     # code_lines = code_lines.reset_index(name="line_number")

    #     with open(Path(source_dir) / source_path, 'r') as f:
    #         lines = f.readlines()
        
    #     code_lines = pd.DataFrame({
    #         'line_number': range(1, len(lines) + 1),
    #         'code': lines,
    #         'dummy_key': 1
    #     })

    #     object_positions = carl_object_positions[carl_object_positions["source_path"] == source_path]
    #     object_positions = object_positions.rename(columns={
    #         "v1": "line_start",
    #         "v2": "col_start",
    #         "v3": "line_end",
    #         "v4": "col_end"
    #     })
    #     object_positions["source_path"] = object_positions["source_path"].str.replace("/home/carl/sources_wrk/", "")

    #     object_positions['dummy_key'] = 1

    #     df = pd.merge(code_lines, object_positions, on="dummy_key", how="inner")
    #     df = df[(df["line_number"] <= df["line_end"]) & (df["line_number"] >= df["line_start"])]
    #     df["line_span"] = df["line_end"] - df["line_start"] + 1

    #     df = (
    #         df.groupby(["source_path", "line_number", "code"])
    #         .apply(lambda x: pd.DataFrame({
    #             "nb_scan_id": [len(x["scan_id"].unique())],
    #             "scan_ids": [",".join(map(str, x["scan_id"].unique()))]
    #         }))
    #         .reset_index()
    #         .sort_values("line_number")
    #         .assign(range_idx=lambda x: x["scan_ids"].ne(x["scan_ids"].shift()).cumsum())
    #         .assign(scan_id=lambda x: x["scan_ids"].str.split(",", expand=True)[0].apply(lambda s: int(s) if s else None))
    #         .assign(parent_scan_id=lambda x: x["scan_ids"].str.split(",", expand=True)[1].apply(lambda s: int(s) if s else None))
    #         [["line_number", "range_idx", "scan_ids", "code", "scan_id", "parent_scan_id", "source_path"]]
    #     )

    #     result.append(df)

    # return pd.concat(result, ignore_index=True)


    def process_file(source_dir, file_path):
        # Read lines from file
        with open(str(source_dir)+'/'+file_path, 'r') as file:
            lines = file.readlines()
        
        # Create a DataFrame for lines with line numbers
        df_lines = pd.DataFrame({
            'line_number': range(1, len(lines) + 1),
            'code': [line.strip() for line in lines]
        })
        
        # Filter and transform carl_object_positions
        carl_filtered = carl_object_positions.copy()
        carl_filtered['source_path'] = carl_filtered['source_path'].str.replace("/home/carl/sources_wrk/", "")
        carl_filtered = carl_filtered[carl_filtered['source_path'] == file_path]
        
        carl_filtered = carl_filtered.assign(
            line_start=carl_filtered['v1'].astype(int),
            col_start=carl_filtered['v2'].astype(int),
            line_end=carl_filtered['v3'].astype(int),
            col_end=carl_filtered['v4'].astype(int)
        )
        
        # Cross join df_lines with carl_filtered
        df_cross = df_lines.assign(key=1).merge(carl_filtered.assign(key=1), on='key').drop('key', axis=1)
        
        # Filter lines within the specified range
        df_filtered = df_cross[(df_cross['line_number'] <= df_cross['line_end']) & (df_cross['line_number'] >= df_cross['line_start'])]
        
        # Add line_span column
        df_filtered['line_span'] = df_filtered['line_end'] - df_filtered['line_start'] + 1
        
        # Group and summarize
        grouped = df_filtered.groupby(['source_path', 'line_number', 'code']).agg(
            nb_scan_id=pd.NamedAgg(column='scan_id', aggfunc='nunique'),
            scan_ids=pd.NamedAgg(column='scan_id', aggfunc=lambda x: '#'.join(map(str, x)))
        ).reset_index()
        
        # Arrange by line_number
        grouped = grouped.sort_values(by='line_number')
        
        # Create range_idx and cumulative sum
        grouped['range_idx'] = grouped['scan_ids'] != grouped['scan_ids'].shift(1)
        grouped['range_idx'] = grouped['range_idx'].cumsum().fillna(1).astype(int)
        
        # Ensure scan_ids is a string before splitting
        grouped['scan_ids'] = grouped['scan_ids'].astype(str)
        scan_ids_split = grouped['scan_ids'].str.split('#', expand=True)
        
        grouped = pd.concat([grouped, scan_ids_split], axis=1)
        grouped = grouped.rename(columns={2: 'scan_id', 1: 'parent_scan_id'})
        
        # Convert scan_id and parent_scan_id to numeric
        grouped['scan_id'] = pd.to_numeric(grouped['scan_id'], errors='coerce')
        grouped['parent_scan_id'] = pd.to_numeric(grouped['parent_scan_id'], errors='coerce')
        
        # Select required columns
        result = grouped[['line_number', 'range_idx', 'scan_id', 'code', 'parent_scan_id', 'source_path']]
        
        return result

    results = pd.concat([process_file(source_dir, path) for path in object_path_list], ignore_index=True)

    return results

def map_pattern_occurrence_object(carl_object_positions, pattern_occurrence_locations, debug_mode=False):
    if debug_mode:
        import pdb; pdb.set_trace()

    print(carl_object_positions['source_path'])    
    
    # Clean source_path in carl_object_positions
    carl_object_positions['source_path'] = carl_object_positions['source_path'].str.replace("/home/carl/sources_wrk/", "")
    
    print(carl_object_positions['source_path'])  

    # Convert relevant columns to integers properly
    carl_object_positions[['v1', 'v2', 'v3', 'v4']] = carl_object_positions[['v1', 'v2', 'v3', 'v4']].apply(lambda x: x.str.split('#').str[0]).astype(int)
    
    # Perform the inner join
    pattern_occurrence_objects = pd.merge(
        carl_object_positions,
        pattern_occurrence_locations.query('req_nat == "BLOCKER"')[['source_path', 'line_number', 'req_name', 'req_id', 'req_nat', 'technology']],
        how='inner',
        on='source_path'
    )

    print(pattern_occurrence_objects)
    
    # Filter and manipulate data
    pattern_occurrence_objects = pattern_occurrence_objects[
        (pattern_occurrence_objects['line_number'].astype(int) >= pattern_occurrence_objects['v1'].astype(int)) &
        (pattern_occurrence_objects['line_number'].astype(int) <= pattern_occurrence_objects['v3'].astype(int))
    ].assign(
        object_loc=lambda x: x['v3'] - x['v1'] + 1
    ).groupby(['source_path', 'req_id', 'line_number']).apply(
        lambda x: x.sort_values(by='object_loc').iloc[0]
    ).reset_index(drop=True)

    pattern_occurrence_objects = pattern_occurrence_objects.rename(columns={'v1': 'line_start', 'v2': 'col_start', 'v3': 'line_end', 'v4':'col_end'})
    # carl_object_positions = carl_object_positions.rename(columns={'v1': 'line_start', 'v2': 'col_start', 'v3': 'line_end', 'v4':'col_end'})
    
    # Assuming get_object_range is another function returning required object ranges
    pattern_occurrence_object_ranges = get_object_range(
        carl_object_positions=carl_object_positions,
        object_path_list=pattern_occurrence_objects['source_path'].unique(),
        # source_dir=carl_object_positions['source_path'],  # Adjust this path accordingly
        debug_mode=debug_mode
    )
    
    pattern_occurrence_object_ranges['parent_scan_id'] = pattern_occurrence_object_ranges['parent_scan_id'].fillna(0).astype(int)

    return {
        'pattern_occurrence_objects': pattern_occurrence_objects,
        'pattern_occurrence_object_ranges': pattern_occurrence_object_ranges
    }

def prepare_code_snippets_hl_insight(source_dir, pattern_occurrence_objects, pattern_occurrence_object_ranges, min_line_span, max_line_span, debug_mode=False):
    if debug_mode:
        import pdb; pdb.set_trace()

    # Ensure columns are of the correct type
    pattern_occurrence_objects['line_number'] = pattern_occurrence_objects['line_number'].astype(int)
    pattern_occurrence_object_ranges['line_number'] = pattern_occurrence_object_ranges['line_number'].astype(int)
    
    pattern_occurrence_object_ranges_unique = pattern_occurrence_object_ranges.drop_duplicates(subset=['scan_id', 'source_path', 'range_idx'])

    snippets = []

    for _, row in pattern_occurrence_object_ranges_unique.iterrows():
        if debug_mode:
            import pdb; pdb.set_trace()

        target_scan_id = row['scan_id']
        target_source_path = row['source_path']
        target_range_idx = row['range_idx']

        obj_range_code = pattern_occurrence_object_ranges[
            (pattern_occurrence_object_ranges['scan_id'] == target_scan_id) &
            (pattern_occurrence_object_ranges['source_path'] == target_source_path) &
            (pattern_occurrence_object_ranges['range_idx'] == target_range_idx)
        ]

        range_max = obj_range_code['line_number'].max()
        range_min = obj_range_code['line_number'].min()
        range_size = range_max - range_min + 1

        obj_findings = pattern_occurrence_objects[
            (pattern_occurrence_objects['source_path'] == target_source_path) &
            (pattern_occurrence_objects['scan_id'] == target_scan_id) &
            (pattern_occurrence_objects['line_number'] >= range_min) &
            (pattern_occurrence_objects['line_number'] <= range_max)
        ][['technology', 'line_number', 'req_id']].drop_duplicates()

        obj_findings = obj_findings.groupby('line_number').agg(lambda x: '||'.join(x.unique())).reset_index()
        obj_findings['flagged'] = 1

        if not obj_findings.empty:
            if range_size <= max_line_span:
                merged = obj_range_code.merge(obj_findings, on='line_number', how='left')
                merged['flagged'] = merged['flagged'].fillna(0)
                merged['technology'] = merged['technology'].fillna("")
                merged['req_id'] = merged['req_id'].fillna("")
                merged['group_idx'] = 1

                summary = merged.groupby('group_idx').agg({
                    'line_number': ['min', 'max'],
                    'technology': lambda x: ''.join(x.unique()),
                    'code': lambda x: '\n'.join(x),
                    'req_id': lambda x: '||'.join(x.unique())
                }).reset_index()

                summary.columns = ['group_idx', 'line_start', 'line_end', 'technology', 'code', 'req_id_list']
                summary['line_span'] = summary['line_end'] - summary['line_start']
                summary['source_path'] = target_source_path
                summary['scan_id'] = target_scan_id
                summary['src'] = 'whole'
                
                snippets.append(summary)
            else:
                merged = obj_range_code.merge(obj_findings, on='line_number', how='left')
                merged['flagged'] = merged['flagged'].fillna(0)
                merged['technology'] = merged['technology'].fillna("")
                merged['req_id'] = merged['req_id'].fillna("")

                merged['near_flagged'] = merged['flagged'].rolling(window=min_line_span, center=True).sum().fillna(0)
                merged['group_line'] = merged['near_flagged'] > 0
                merged['group_idx'] = (merged['group_line'] != merged['group_line'].shift()).cumsum()
                
                merged['process_group'] = merged.groupby('group_idx')['flagged'].transform('any')
                filtered = merged[merged['process_group']]

                summary = filtered.groupby('group_idx').agg({
                    'line_number': ['min', 'max'],
                    'technology': lambda x: ''.join(x.unique()),
                    'code': lambda x: '\n'.join(x),
                    'req_id': lambda x: '||'.join(x.unique())
                }).reset_index()

                summary.columns = ['group_idx', 'line_start', 'line_end', 'technology', 'code', 'req_id_list']
                summary['line_span'] = summary['line_end'] - summary['line_start']
                summary['source_path'] = target_source_path
                summary['scan_id'] = target_scan_id
                summary['src'] = 'window'
                
                snippets.append(summary)

    pattern_occurrence_snippets = pd.concat(snippets, ignore_index=True)

    pattern_occurrence_snippets['scan_id'] = pattern_occurrence_snippets['scan_id'].astype(int)
    
    return {
        'pattern_occurrence_snippets': pattern_occurrence_snippets,
        'pattern_occurrence_objects': pattern_occurrence_objects
    }

def prepare_code_snippets_carl_path_insight(
    source_dir: str,
    pattern_occurrence_objects: pd.DataFrame,
    pattern_occurrence_object_ranges: pd.DataFrame,
    min_line_span: int,
    max_line_span: int,
    debug_mode: bool = False
) -> Dict[str, pd.DataFrame]:
    
    if debug_mode:
        import pdb; pdb.set_trace()
    
    # Function to join based on source path
    def join_by(df, col):
        return df[col]

    # Step 1: Adjust source path and join with locations
    pattern_occurrence_objects = (
        pattern_occurrence_objects.rename(columns={
            'source_path': 'source_path_original',
            'v1': 'line_start',
            'v2': 'col_start',
            'v3': 'line_end',
            'v4': 'col_end'
        })
        .assign(
            source_path=lambda x: x['source_path_original'].str.replace("/home/carl/sources_wrk/", "")
        )
        .merge(
            pattern_occurrence_locations[['source_path', 'line_number', 'end_line_number', 'req_name', 'req_id', 'occurrence_id', 'step']],
            on='source_path'
        )
        .query('line_number >= line_start and end_line_number <= line_end')
        .assign(object_loc=lambda x: x['line_end'] - x['line_start'] + 1)
        .groupby(['source_path', 'req_id', 'line_number'])
        .apply(lambda x: x.sort_values(by='object_loc').iloc[0])
        .reset_index(drop=True)
    )

    # Step 2: Prepare pattern_occurrence_snippets
    def process_snippets(req_name, req_id, req_data):
        result = []
        for target_occurrence_id, occurrence_data in req_data.iterrows():
            df = (
                occurrence_data.merge(pattern_occurrence_objects.query(f'req_id == {req_id}'))
                .groupby(['scan_id', 'fullname', 'name', 'techno', 'type_str', 'source_path', 'line_start', 'line_end', 'line_number', 'end_line_number'])
                .agg({'occurrence_id': 'nunique'})
                .reset_index()
                .assign(occurrence_id=lambda x: target_occurrence_id)
            )
            result.append(df)
        return pd.concat(result, ignore_index=True).assign(req_id=req_id, req_name=req_name)

    pattern_occurrence_snippets = (
        pattern_occurrence_objects
        .groupby(['req_name', 'req_id', 'occurrence_id'])
        .apply(lambda x: process_snippets(x.name, x['req_id'].iloc[0], x))
        .reset_index(drop=True)
    )

    # Step 3: Prepare pattern_occurrence_snippet_paths
    def process_snippet_paths(target_scan_id, target_source_path):
        if debug_mode:
            import pdb; pdb.set_trace()

        flagged_file = target_source_path
        if not os.path.exists(flagged_file):
            flagged_file = os.path.join(source_dir, target_source_path)
        if not os.path.exists(flagged_file):
            raise FileNotFoundError(f"File '{target_source_path}' not found.")
        
        obj_findings = (
            pattern_occurrence_objects
            .query(f'source_path == "{target_source_path}" and scan_id == {target_scan_id}')
            .groupby('line_number')
            .agg(lambda x: '||'.join(pd.unique(x)))
            .reset_index()
            .assign(flagged=1)
        )

        obj_positions = (
            pattern_occurrence_objects
            .query(f'source_path == "{target_source_path}" and scan_id == {target_scan_id}')
            .loc[:, ['line_start', 'line_end', 'source_path']]
            .drop_duplicates()
        )

        flagged_content = (
            pd.Series(pd.read_lines(flagged_file), name='code')
            .reset_index(name='line_number')
            .query(f'line_number >= {obj_positions["line_start"]} and line_number <= {obj_positions["line_end"]}')
        )

        flagged_content = (
            flagged_content
            .merge(obj_findings, on='line_number', how='left')
            .fillna({'flagged': 0, 'req_id': ''})
            .assign(near_flagged=lambda x: np.convolve(x['flagged'], np.ones(min_line_span), mode='same'))
            .assign(group_line=lambda x: x['near_flagged'] > 0)
            .assign(group_idx=lambda x: (x['group_line'] != x['group_line'].shift()).astype(int).cumsum())
            .groupby(['group_idx', 'technology'])
            .filter(lambda x: any(x['flagged'] > 0))
            .assign(
                line_start=lambda x: x['line_number'].min(),
                line_end=lambda x: x['line_number'].max(),
                code=lambda x: '\n'.join(x['code'])
            )
            .groupby(['group_idx', 'technology', 'line_start', 'line_end'])
            .agg(req_id_list=('req_id', lambda x: '||'.join(pd.unique(x))),
                 occurrence_id_list=('occurrence_id', lambda x: '||'.join(pd.unique(x))))
            .reset_index()
            .assign(source_path=target_source_path,
                    scan_id=target_scan_id,
                    file_path=flagged_file,
                    line_span=lambda x: x['line_end'] - x['line_start'])
        )

        return flagged_content

    pattern_occurrence_snippet_paths = (
        pattern_occurrence_objects
        .loc[:, ['scan_id', 'source_path']]
        .drop_duplicates()
        .apply(lambda x: process_snippet_paths(x['scan_id'], x['source_path']), axis=1)
    )

    # Step 4: Prepare pattern_occurrence_code
    def process_code(target_scan_id, target_source_path):
        if debug_mode:
            import pdb; pdb.set_trace()

        flagged_file = target_source_path
        if not os.path.exists(flagged_file):
            flagged_file = os.path.join(source_dir, target_source_path)
        if not os.path.exists(flagged_file):
            raise FileNotFoundError(f"File '{target_source_path}' not found.")

        obj_findings = (
            pattern_occurrence_objects
            .query(f'source_path == "{target_source_path}" and scan_id == {target_scan_id}')
            .loc[:, ['req_id', 'occurrence_id', 'line_start', 'line_end', 'source_path', 'technology']]
            .drop_duplicates()
            .groupby(['line_start', 'line_end', 'source_path', 'technology'])
            .agg(lambda x: '||'.join(pd.unique(x)))
            .reset_index()
            .assign(group_idx=1)
        )

        obj_positions = (
            pattern_occurrence_objects
            .query(f'source_path == "{target_source_path}" and scan_id == {target_scan_id}')
            .loc[:, ['line_start', 'line_end', 'source_path']]
            .drop_duplicates()
        )

        flagged_content = (
            pd.Series(pd.read_lines(flagged_file), name='code')
            .reset_index(name='line_number')
            .query(f'line_number >= {obj_positions["line_start"]} and line_number <= {obj_positions["line_end"]}')
            .assign(code=lambda x: '\n'.join(x['code']))
        )

        return (
            obj_findings
            .merge(flagged_content, on=['line_start', 'line_end', 'source_path'])
            .assign(
                scan_id=target_scan_id,
                source_path=target_source_path,
                line_span=lambda x: x['line_end'] - x['line_start']
            )
        )

    pattern_occurrence_code = (
        pattern_occurrence_objects
        .loc[:, ['scan_id', 'source_path']]
        .drop_duplicates()
        .apply(lambda x: process_code(x['scan_id'], x['source_path']), axis=1)
    )

    # Additional processing
    pattern_occurrence_snippets['nb_req'] = pattern_occurrence_snippets['req_id_list'].str.count('\\|\\|') + 1
    pattern_occurrence_snippets['nb_occurrence'] = pattern_occurrence_snippets['occurrence_id_list'].str.count('\\|\\|') + 1

    pattern_occurrence_code['nb_req'] = pattern_occurrence_code['req_id_list'].str.count('\\|\\|') + 1
    pattern_occurrence_code['nb_occurrence'] = pattern_occurrence_code['occurrence_id_list'].str.count('\\|\\|') + 1

    pattern_occurrence_code_paths = (
        pattern_occurrence_objects
        .groupby(['req_name', 'req_id', 'occurrence_id'])
        .apply(lambda x: pd.DataFrame({
            'scan_id': x['scan_id'].iloc[0],
            'source_path': x['source_path'].iloc[0],
            'fullname': x['fullname'].iloc[0],
            'name': x['name'].iloc[0],
            'techno': x['techno'].iloc[0],
            'type_str': x['type_str'].iloc[0],
            'scan_path': x['scan_path'].iloc[0],
            'line_start': x['line_start'].iloc[0],
            'line_end': x['line_end'].iloc[0],
            'code': pattern_occurrence_code.query(f'req_name == "{x.name[0]}" and req_id == {x.name[1]} and occurrence_id == {x.name[2]}')['code'].tolist(),
            'req_name': x.name[0],
            'req_id': x.name[1],
            'occurrence_id': x.name[2],
            'object_loc': x['object_loc'].iloc[0],
            'nb_req': x['req_id_list'].str.count('\\|\\|').iloc[0] + 1,
            'nb_occurrence': x['occurrence_id_list'].str.count('\\|\\|').iloc[0] + 1,
        }))
        .reset_index(drop=True)
    )

    pattern_occurrence_snippet_paths = (
        pattern_occurrence_objects
        .groupby(['req_name', 'req_id', 'occurrence_id'])
        .apply(lambda x: pd.DataFrame({
            'group_idx': x['group_idx'].iloc[0],
            'scan_id': x['scan_id'].iloc[0],
            'source_path': x['source_path'].iloc[0],
            'fullname': x['fullname'].iloc[0],
            'name': x['name'].iloc[0],
            'techno': x['techno'].iloc[0],
            'type_str': x['type_str'].iloc[0],
            'scan_path': x['scan_path'].iloc[0],
            'line_start': x['line_start'].iloc[0],
            'line_end': x['line_end'].iloc[0],
            'code': pattern_occurrence_snippets.query(f'group_idx == {x["group_idx"].iloc[0]}')['code'].tolist(),
            'req_name': x.name[0],
            'req_id': x.name[1],
            'occurrence_id': x.name[2],
            'object_loc': x['object_loc'].iloc[0],
            'nb_req': x['req_id_list'].str.count('\\|\\|').iloc[0] + 1,
            'nb_occurrence': x['occurrence_id_list'].str.count('\\|\\|').iloc[0] + 1,
        }))
        .reset_index(drop=True)
    )

    return {
        'pattern_occurrence_snippets': pattern_occurrence_snippets,
        'pattern_occurrence_snippet_paths': pattern_occurrence_snippet_paths,
        'pattern_occurrence_code': pattern_occurrence_code,
        'pattern_occurrence_code_paths': pattern_occurrence_code_paths,
        'pattern_occurrence_objects': pattern_occurrence_objects
    }

def prepare_code_snippets_carl_insight(
    source_dir,
    pattern_occurrence_objects,
    pattern_occurrence_object_ranges,
    min_line_span,
    max_line_span,
    debug_mode=False):
    
    if debug_mode:
        import pdb; pdb.set_trace()  # Python's equivalent to R's browser()

    # Add your code snippet preparation logic here
    # For now, it's just a placeholder
    
    pass  # Placeholder for actual code logic

def build_carl_from_to(carl_object_callees, carl_object_projects, debug_mode=False):
    if debug_mode:
        import pdb; pdb.set_trace()

    # Ensure the required columns are strings
    carl_object_callees['v1'] = carl_object_callees['v1'].astype(str)
    carl_object_callees['v3'] = carl_object_callees['v3'].astype(str)
    carl_object_callees['link_type_str'] = carl_object_callees['link_type_str'].astype(str)

    carl_object_callees['nb_bookmarks'] = 1 + carl_object_callees['v1'].str.count('#')
    carl_object_callees['line_starts'] = carl_object_callees['v1']
    carl_object_callees['line_ends'] = carl_object_callees['v3']

    carl_object_callees = carl_object_callees.rename(columns={
        'scan_id': 'from',
        'source_path': 'from_path',
        'type_str': 'from_type',
        'link_type': 'link_type_id',
        'link_type_str': 'link_type'
    })

    carl_object_projects = carl_object_projects.rename(columns={
        'project': 'callee_project',
        'scan_id': 'callee',
        'source_path': 'callee_path',
        'type_str': 'callee_type'
    })

    merged_df = carl_object_callees.merge(
        carl_object_projects[['callee_project', 'callee', 'callee_path', 'callee_type']].dropna(),
        on=['callee', 'callee_project'],
        how='left',
        suffixes=('', '_proj')
    )

    # print(merged_df.columns.duplicated())

    # Ensure the merged DataFrame does not have duplicate columns
    merged_df = merged_df.loc[:, ~merged_df.columns.duplicated()]

    carl_from_to = (
        merged_df.dropna()
        .assign(
            from_path=lambda df: df['from_path'].str.replace(r'^/home/carl/(sources|output|sources_wrk|output_wrk)/', '', regex=True),
            to=lambda df: df['callee'],
            to_type=lambda df: df['callee_type'],
            to_project=lambda df: df['callee_project'],
            to_path=lambda df: df['callee_path'].str.replace(r'^/home/carl/(sources|output|sources_wrk|output_wrk)/', '', regex=True)
        )
        .groupby(['from', 'from_path', 'from_type', 'to', 'to_project', 'to_path', 'to_type', 'link_type'])
        .agg(
            nb_bookmarks=('nb_bookmarks', 'sum'),
            line_starts=('line_starts', lambda x: '#'.join(x)),
            line_ends=('line_ends', lambda x: '#'.join(x))
        )
        .reset_index()
    )

    return {'carl_from_to': carl_from_to}

def build_carl_exceptions(carl_from_to, carl_object_projects, debug_mode=False):
    if debug_mode:
        import pdb; pdb.set_trace()
    
    # Ensure all values in 'link_type' are strings
    carl_from_to['link_type'] = carl_from_to['link_type'].astype(str)
    
    # Filter rows where 'link_type' contains 'raise', 'throw', or 'catch'
    carl_exceptions = carl_from_to[carl_from_to['link_type'].str.contains("(raise|throw|catch)", regex=True)]
    
    # Merge with carl_object_projects on specified columns
    carl_exceptions = carl_exceptions.merge(
        carl_object_projects,
        left_on=['to', 'to_path', 'to_type'],
        right_on=['scan_id', 'source_path', 'type_str']
    )
    
    # Transform the merged DataFrame
    carl_exceptions = carl_exceptions.assign(
        scan_id=carl_exceptions['from'],
        source_path=carl_exceptions['from_path'],
        link_type=carl_exceptions['link_type'].str.replace("Link$", "", regex=True),
        exception=carl_exceptions['fullname']
    )[['scan_id', 'source_path', 'link_type', 'exception', 'external']]
    
    return {'carl_exceptions': carl_exceptions}

# def build_carl_signatures(carl_object_args, carl_object_positions, debug_mode=False):
#     if debug_mode:
#         import pdb; pdb.set_trace()

#     # Calculate sig_starts
#     sig_starts = (
#         carl_object_args[carl_object_args['arg_position'] > 0]
#         .sort_values(by=['scan_id', 'source_path', 'type_str', 'name', 'arg_position'])
#         .groupby(['scan_id', 'source_path', 'type_str', 'name'])
#         .agg(args_list=('arg_type', lambda x: ",".join(x)))
#         .reset_index()
#     )

#     sig_starts['sig_start'] = sig_starts.apply(lambda row: f"{row['name']}({row['args_list']})", axis=1)
#     sig_starts = sig_starts[['scan_id', 'source_path', 'type_str', 'sig_start']]
#     # Perform the anti-join
#     missing_sig_starts = pd.merge(
#         carl_object_positions,
#         sig_starts[['scan_id', 'source_path']],
#         on=['scan_id', 'source_path'],
#         how='left',
#         indicator=True
#     ).query('_merge == "left_only"').drop('_merge', axis=1)
#     # Create the sig_start column
#     missing_sig_starts['sig_start'] = missing_sig_starts['name'] + "()"
#     # Select the required columns
#     missing_sig_starts = missing_sig_starts[['scan_id', 'source_path', 'type_str', 'sig_start']]
#     # Calculate sig_ends
#     sig_ends = (carl_object_args
#                 .query('arg_position == -1')
#                 .assign(sig_end=lambda df: 'return ' + df['arg_type'])
#                 [['scan_id', 'source_path', 'sig_end']])
#     # Join sig_starts and sig_ends
#     carl_signatures = (pd.concat([sig_starts[['scan_id', 'source_path', 'type_str', 'sig_start']],
#                                   missing_sig_starts[['scan_id', 'source_path', 'type_str', 'sig_start']]],
#                                  ignore_index=True)
#                        .merge(sig_ends, on=['scan_id', 'source_path'], how='left')
#                        .assign(source_path=lambda df: df['source_path'].str.replace('^/home/carl/(sources|output|sources_wrk|output_wrk)/', ''))
#                        .assign(signature=lambda df: df['sig_start'].str.cat(df['sig_end'], sep=' ').str.replace('^NA ', '').str.replace(' NA$', ''))
#                        .drop(columns=['sig_start', 'sig_end'])
#                        .drop_duplicates())
#     return {'carl_signatures': carl_signatures}

def build_carl_signatures(carl_object_args, carl_object_positions, debug_mode=False):
    if debug_mode:
        import pdb; pdb.set_trace()
    
    # Creating sig_starts
    sig_starts = (
        carl_object_args[carl_object_args['arg_position'] > 0]
        .sort_values(by='arg_position')
        .groupby(['scan_id', 'source_path', 'type_str', 'name'])
        .agg(args_list=('arg_type', lambda x: ','.join(x)))
        .reset_index()
        .assign(sig_start=lambda df: df['name'] + '(' + df['args_list'] + ')')
    )
    
    # Creating missing_sig_starts
    # merged = pd.merge(carl_object_positions, sig_starts, on=['scan_id', 'source_path'], how='left', indicator=True)
    # missing_sig_starts = merged[merged['_merge'] == 'left_only'].drop(columns='_merge')
    missing_sig_starts = pd.merge(
        carl_object_positions,
        sig_starts[['scan_id', 'source_path']],
        on=['scan_id', 'source_path'],
        how='left',
        indicator=True
    ).query('_merge == "left_only"').drop('_merge', axis=1)
    missing_sig_starts['sig_start'] = missing_sig_starts['name'] + '()'
    
    # Creating sig_ends
    sig_ends = (
        carl_object_args[carl_object_args['arg_position'] == -1]
        .assign(sig_end=lambda df: 'return ' + df['arg_type'])
        [['scan_id', 'source_path', 'sig_end']]
    )
    
    # Combining results
    sig_combined = pd.concat([sig_starts[['scan_id', 'source_path', 'type_str', 'sig_start']], 
                              missing_sig_starts[['scan_id', 'source_path', 'type_str', 'sig_start']]])
    
    carl_signatures = pd.merge(sig_combined, sig_ends, on=['scan_id', 'source_path'], how='left')
    
    carl_signatures['source_path'] = carl_signatures['source_path'].str.replace(r'^/home/carl/(sources|output|sources_wrk|output_wrk)/', '')
    carl_signatures['signature'] = carl_signatures['sig_start'] + ' ' + carl_signatures['sig_end'].fillna('')
    carl_signatures['signature'] = carl_signatures['signature'].str.replace(r'^NA ', '').str.replace(r' NA$', '')
    
    carl_signatures = carl_signatures.drop(columns=['sig_start', 'sig_end']).drop_duplicates()
    
    return {'carl_signatures': carl_signatures}

def direct_caller_census(carl_from_to, carl_signatures, transformation_objects, debug_mode=False):
    if debug_mode:
        import pdb; pdb.set_trace()
    
    transformation_direct_impact = pd.merge(
        transformation_objects,
        carl_from_to,
        left_on=['to', 'to_path'],
        right_on=['to', 'to_path'],
        how='inner'
    )

    transformation_isolated = pd.merge(
        transformation_objects, 
        carl_from_to,
        how='outer',
        on=['to', 'to_path'],
        indicator=True
    )

    transformation_isolated = transformation_isolated[
        transformation_isolated['_merge'] == 'left_only'
    ].drop('_merge', axis=1)
    
    # Group by and summarise
    transformation_direct_impact_stats = (
        transformation_direct_impact
        .groupby(['to', 'to_path', 'from_type', 'link_type'])
        .size()
        .reset_index(name='nb_impact')
        .rename(columns={'to':'scan_id', 'to_path':'source_path'})
    )
        

    # Merge dataframes
    merged_df = pd.merge(transformation_direct_impact,
                        carl_signatures.rename(columns={'signature': 'to_signature', 'type_str': 'to_type'}),
                        left_on=['to', 'to_path', 'to_type'],
                        right_on=['scan_id', 'source_path', 'to_type'])

    # Function to process each row
    def process_row(row):
        if debug_mode:
            print("Debug mode active, inspecting row:")
            print(row)
        
        line_starts = row['line_starts'].split("#")
        line_ends = row['line_ends'].split("#")
        
        caller_lines = pd.DataFrame({'bk_start': line_starts, 'bk_end': line_ends})
        
        abs_source_path = source_dir / row['from_path']
        
        if abs_source_path.exists():
            with open(abs_source_path, 'r') as file:
                content = file.readlines()
            
            content_df = pd.DataFrame(content, columns=['code'])
            content_df['line_number'] = content_df.index + 1
            
            bk_code_list = []
            
            for bk_start, bk_end in zip(line_starts, line_ends):
                bk_start, bk_end = int(bk_start), int(bk_end)
                bk_code = "\n".join(content_df[(content_df['line_number'] >= bk_start) & (content_df['line_number'] <= bk_end)]['code'])
                bk_code_list.append({'bk_code': bk_code, 'bk_idx': int(bk_start), 'bk_span': bk_end - bk_start + 1})
            
            bk_code_df = pd.DataFrame(bk_code_list)
            
            bk_code_df['to'] = row['to']
            bk_code_df['to_path'] = row['to_path']
            bk_code_df['to_signature'] = row['to_signature']
            bk_code_df['to_project'] = row['to_project']
            bk_code_df['to_type'] = row['to_type']
            bk_code_df['from'] = row['from']
            bk_code_df['from_path'] = row['from_path']
            bk_code_df['from_type'] = row['from_type']
            bk_code_df['link_type'] = row['link_type']
            bk_code_df['bk_line_starts'] = row['line_starts']
            bk_code_df['bk_line_ends'] = row['line_ends']
            
            return bk_code_df

    # Process each row
    transformation_direct_impact_code_long = pd.concat([process_row(row) for _, row in merged_df.iterrows()])

      
    if not transformation_direct_impact_code_long.empty:
        transformation_direct_impact_code = (
            transformation_direct_impact_code_long
            .groupby(['to', 'to_path', 'to_type', 'to_signature', 'to_project'])
            .apply(lambda x: pd.Series({
                'from_type_list': list(x['from_type']),
                'from_list': list(x['from']),
                'from_path_list': list(x['from_path']),
                'bk_code_list': list(x['bk_code']),
                'bk_idx_list': list(x['bk_idx']),
                'bk_line_starts_list': list(x['bk_line_starts']),
                'bk_line_ends_list': list(x['bk_line_ends']),
                'bk_span_list': list(x['bk_span']),
                'link_type_list': list(x['link_type'])
            }))
            .reset_index()
        )
    else:
        transformation_direct_impact_code = None
    
    return {
        'transformation_direct_impact_stats': transformation_direct_impact_stats,
        'transformation_direct_impact_code': transformation_direct_impact_code,
        'transformation_direct_impact_code_long': transformation_direct_impact_code_long,
        'transformation_direct_impact': transformation_direct_impact,
        'transformation_isolated': transformation_isolated
    }

def enclosed_census(transformation_object_ranges, transformation_objects, debug_mode=False):
    if debug_mode:
        import pdb; pdb.set_trace()

    if "fullname" not in transformation_objects.columns:
        transformation_objects = (
            carl_object_positions
            .assign(
                source_path=lambda df: df['source_path'].str.replace('/home/carl/sources_wrk/', ''),
                line_start=lambda df: df['v1'].astype(int),
                col_start=lambda df: df['v2'].astype(int),
                line_end=lambda df: df['v3'].astype(int),
                col_end=lambda df: df['v4'].astype(int)
            )
            .rename(columns={
                'scan_id': 'scan_id',
                'fullname': 'fullname',
                'name': 'name',
                'techno': 'techno',
                'type_str': 'type_str',
                'source_path': 'source_path',
                'line_start': 'line_start',
                'col_start': 'col_start',
                'line_end': 'line_end',
                'col_end': 'col_end'
            })
            .merge(transformation_objects, on=['scan_id', 'source_path'], how='inner')
            .assign(object_loc=lambda df: df['line_end'] - df['line_start'] + 1)
        )

    occurrence_parent_graph = (
        transformation_object_ranges[['parent_scan_id', 'scan_id']]
        .drop_duplicates()
        .dropna()
        .applymap(str)
    )

    graph = nx.from_pandas_edgelist(occurrence_parent_graph, 'parent_scan_id', 'scan_id', create_using=nx.DiGraph())
    graph_nodes = set(graph.nodes)

    def local_neighborhood(scan_id):
        if str(scan_id) not in graph_nodes:
            return pd.DataFrame({'scan_id': [], 'to': []})  # Return an empty DataFrame if scan_id is not in the graph
        neighborhood = nx.single_source_shortest_path_length(graph, str(scan_id), cutoff=5)
        return pd.DataFrame({'scan_id': scan_id, 'to': neighborhood.keys()})

    transformation_enclosed_long = (
        pd.concat([local_neighborhood(scan_id) for scan_id in transformation_objects['scan_id']], ignore_index=True)
        .astype(int)
        .query('scan_id != to')
        .merge(transformation_objects, on='scan_id', how='inner')
    )

    if not transformation_enclosed_long.empty:
        transformation_enclosed_code_long = (
            transformation_enclosed_long
            .merge(
                transformation_object_ranges[['source_path', 'line_number', 'scan_id', 'code']].rename(columns={'scan_id': 'to', 'line_number': 'to_line_number'}),
                on=['to', 'source_path'],
                how='inner'
            )
            .sort_values(by='to_line_number')
            .groupby(
                [
                    'scan_id', 'fullname', 'name', 'techno', 'type_str', 'source_path',
                    'line_start', 'line_end', 'col_start', 'col_end', 'line_number',
                    'req_name', 'req_id', 'req_nat', 'technology', 'object_loc', 'to'
                ],
                as_index=False
            )
            .agg(to_code=('code', lambda codes: "\n".join(codes)))
        )

        transformation_enclosed = (
            transformation_enclosed_long
            .groupby(
                [
                    'scan_id', 'fullname', 'name', 'techno', 'type_str', 'source_path',
                    'line_start', 'line_end', 'col_start', 'col_end', 'line_number',
                    'req_name', 'req_id', 'req_nat', 'technology', 'object_loc'
                ],
                as_index=False
            )
            .agg(to_list=('to', list))
        )
    else:
        transformation_enclosed = None
        transformation_enclosed_code_long = pd.DataFrame()

    if not transformation_enclosed_code_long.empty:
        transformation_enclosed_code = (
            transformation_enclosed_code_long
            .groupby(
                [
                    'scan_id', 'fullname', 'name', 'techno', 'type_str', 'source_path',
                    'line_start', 'line_end', 'col_start', 'col_end', 'line_number',
                    'req_name', 'req_id', 'req_nat', 'technology', 'object_loc'
                ],
                as_index=False
            )
            .agg(to_list=('to', list), to_code_list=('to_code', list))
        )
    else:
        transformation_enclosed_code = None

    return {
        'transformation_enclosed': transformation_enclosed,
        'transformation_enclosed_long': transformation_enclosed_long,
        'transformation_enclosed_code': transformation_enclosed_code,
        'transformation_enclosed_code_long': transformation_enclosed_code_long
    }

def build_carl_graph(carl_from_to, pattern_occurrence_objects, carl_signatures, carl_object_projects, pattern_occurrence_object_ranges, debug_mode=False):
    if debug_mode:
        import pdb; pdb.set_trace()
    
    carl_from_to['from'] = carl_from_to['from'].astype(str)
    carl_from_to['to'] = carl_from_to['to'].astype(str)
    
    program_flow_graph = nx.from_pandas_edgelist(
        carl_from_to, 'from', 'to', edge_attr=['link_type', 'nb_bookmarks'], create_using=nx.DiGraph()
    )
    
    alerts_df = pattern_occurrence_objects.groupby(pattern_occurrence_objects['scan_id'].astype(str))['req_name'].nunique().reset_index()
    alerts_df.columns = ['name', 'nb_alerts']
    alerts_df.set_index('name', inplace=True)
    
    nx.set_node_attributes(program_flow_graph, alerts_df['nb_alerts'].to_dict(), name='nb_alerts')
    
    signature_df = carl_signatures[['scan_id', 'signature', 'type_str']].copy()
    signature_df['name'] = signature_df['scan_id'].astype(str)
    signature_df.set_index('name', inplace=True)
    nx.set_node_attributes(program_flow_graph, signature_df[['signature', 'type_str']].to_dict('index'))
    
    carl_object_projects['name'] = carl_object_projects['scan_id'].astype(str)
    project_attrs = carl_object_projects.groupby('name').agg(lambda x: '#'.join(sorted(map(str, x.unique())))).reset_index()
    project_attrs.set_index('name', inplace=True)
    nx.set_node_attributes(program_flow_graph, project_attrs.to_dict('index'))
    
    occurrence_parent_graph_df = pattern_occurrence_object_ranges[['parent_scan_id', 'scan_id']].dropna().drop_duplicates()
    occurrence_parent_graph_df['from'] = occurrence_parent_graph_df['parent_scan_id'].astype(str)
    occurrence_parent_graph_df['to'] = occurrence_parent_graph_df['scan_id'].astype(str)
    occurrence_parent_graph_df['link_type'] = 'defineLink'
    
    occurrence_parent_graph = nx.from_pandas_edgelist(
        occurrence_parent_graph_df, 'from', 'to', edge_attr=['link_type'], create_using=nx.DiGraph()
    )
    
    nx.set_node_attributes(occurrence_parent_graph, alerts_df['nb_alerts'].to_dict(), name='nb_alerts')
    nx.set_node_attributes(occurrence_parent_graph, signature_df[['signature', 'type_str']].to_dict('index'))
    nx.set_node_attributes(occurrence_parent_graph, project_attrs.to_dict('index'))
    
    tainted_node_names = [node for node, attr in program_flow_graph.nodes(data=True) if attr.get('nb_alerts', 0) > 0]
    
    def get_local_neighborhood(graph, node, order, direction):
        neighbors = {node}
        for _ in range(order):
            if direction == 'in':
                neighbors.update(*(set(graph.predecessors(n)) for n in neighbors if n in graph))
            elif direction == 'out':
                neighbors.update(*(set(graph.successors(n)) for n in neighbors if n in graph))
        return neighbors

    tainted_callee_callers = []
    for node in tainted_node_names:
        neighborhood = get_local_neighborhood(program_flow_graph, node, 10, 'in')
        subgraph = program_flow_graph.subgraph(neighborhood)
        for n in subgraph:
            try:
                subgraph.nodes[n]['dist'] = nx.shortest_path_length(subgraph, node, n)
            except nx.NetworkXNoPath:
                subgraph.nodes[n]['dist'] = float('inf')
        tainted_callee_callers.append(subgraph)

    tainted_callee_graphs = nx.compose_all(tainted_callee_callers)
    
    tainted_parent_enclosed = []
    for node in tainted_node_names:
        neighborhood = get_local_neighborhood(occurrence_parent_graph, node, 10, 'out')
        subgraph = occurrence_parent_graph.subgraph(neighborhood)
        for n in subgraph:
            try:
                subgraph.nodes[n]['dist'] = nx.shortest_path_length(subgraph, node, n)
            except nx.NetworkXNoPath:
                subgraph.nodes[n]['dist'] = float('inf')
        tainted_parent_enclosed.append(subgraph)
    
    tainted_parent_graphs = nx.compose_all(tainted_parent_enclosed)
    
    tainted_focus_graph = nx.compose(tainted_parent_graphs, tainted_callee_graphs)
    
    for node, data in tainted_focus_graph.nodes(data=True):
        if 'targets' not in data:
            data['targets'] = ""
        if 'ancestors' not in data:
            data['ancestors'] = ""
    
    for u, v, data in tainted_focus_graph.edges(data=True):
        for key in data:
            if isinstance(data[key], (int, float)) and np.isnan(data[key]):
                data[key] = 0
    
    program_flow_graph_close_focus = tainted_focus_graph.copy()
    
    return {
        'program_flow_graph': program_flow_graph,
        'program_flow_graph_close_focus': program_flow_graph_close_focus
    }

# Getting the documentation
carl_documentation = carl_document_patterns(carl_output_dir, debug_mode=False)

carl_pattern_req_instructions = carl_documentation["carl_pattern_req_instructions"]
carl_blocker_ids = carl_documentation["carl_blocker_ids"]
carl_booster_ids = carl_documentation["carl_booster_ids"]

print(f"carl_blocker_ids: {carl_blocker_ids}")
print(f"carl_booster_ids: {carl_booster_ids}")
print(f"carl_pattern_req_instructions: {carl_pattern_req_instructions}")

# Define pattern_req_instructions based on transformation_source
if transformation_source in ['cloud', 'green']:
    pattern_req_instructions = hl_pattern_req_instructions
elif transformation_source == 'path':
    pattern_req_instructions = carl_pattern_req_instructions
else:
    raise ValueError("Invalid transformation_source")

# Define pattern_req_processing DataFrame
pattern_req_processing_data = {
    'req_id': ["Use_HTTPProtocol", "Use_File_System"],
    'custom_instruction': ["use BASE_URL environment variable value instead", "use 'my_bucket' AWS S3 bucket instead"],
    'technology': ["Java", "Java"]
}

pattern_req_processing = pd.DataFrame(pattern_req_processing_data)

pattern_req_processing_parallel = 1
print(f"pattern_req_processing_parallel: {pattern_req_processing_parallel}")
irun = 0

while True:
    irun += 1
    print(f"\n\n\n\n irun: {irun}\n\n\n")

    if transformation_source == "cloud":
        hl_results = hl_process_cloud_log_files(hl_output_dir=hl_output_dir, hl_cloud_blocker_ids=hl_cloud_blocker_ids, debug_mode=debug_flow)
        
        cloud_log_files = hl_results["cloud_log_files"]
        cloud_log_content = hl_results["cloud_log_content"]
        cloud_pattern_occurrence_locations = hl_results["cloud_pattern_occurrence_locations"]
        cloud_pattern_occurrence_snippets = hl_results["cloud_pattern_occurrence_snippets"]

        print(f"cloud_log_files: {cloud_log_files}")
        print(f"cloud_log_content: {cloud_log_content}")
        print(f"cloud_pattern_occurrence_locations: {cloud_pattern_occurrence_locations}")
        print(f"cloud_pattern_occurrence_snippets: {cloud_pattern_occurrence_snippets}")
        
        # Define the filter function
        def filter_source_path(df):
            return df[~df['source_path'].str.contains("/sourceDir/DLL|/sourceDir/lib|/.mvn/")]

        # Apply the filter function
        cloud_pattern_occurrence_snippets = filter_source_path(cloud_pattern_occurrence_snippets)
        cloud_pattern_occurrence_locations = filter_source_path(cloud_pattern_occurrence_locations)
        pass

    elif transformation_source == "green":
        hl_results = hl_process_green_log_files(hl_output_dir=hl_output_dir, debug_mode=debug_flow)
        
        green_log_files = hl_results['green_log_files']
        green_log_content = hl_results['green_log_content']
        green_pattern_occurrence_locations = hl_results['green_pattern_occurrence_locations']
        green_pattern_occurrence_snippets = hl_results['green_pattern_occurrence_snippets']
        green_url_docs = hl_results['green_url_docs']
        
        green_pattern_occurrence_snippets = green_pattern_occurrence_snippets[
            ~green_pattern_occurrence_snippets['source_path'].str.contains("/sourceDir/DLL|/sourceDir/lib|/.mvn/")
        ]
        
        green_pattern_occurrence_locations = green_pattern_occurrence_locations[
            ~green_pattern_occurrence_locations['source_path'].str.contains("/sourceDir/DLL|/sourceDir/lib|/.mvn/")
        ]

    else:
        carl_results = carl_process_file_results(carl_output_dir=carl_output_dir, debug_mode=debug_flow)
        
        for name in carl_results.keys():
            globals()[name] = carl_results[name].assign(
                source_path=lambda df: df['source_path'].str.replace("/home/carl/(sources|output|sources_wrk|output_wrk)/", "")
            )
            print(globals()[name])
        
        carl_results = carl_process_path_results(carl_output_dir=carl_output_dir, debug_mode=debug_flow)
        
        for name in carl_results.keys():
            globals()[name] = carl_results[name].assign(
                source_path=lambda df: df['source_path'].str.replace("/home/carl/(sources|output|sources_wrk|output_wrk)/", "")
            )
            print(globals()[name])
        
    # Process carl_results from carl_process_metamodel
    carl_results = carl_process_metamodel(carl_output_dir=carl_output_dir, debug_mode=debug_flow)
    carl_types = carl_results["carl_types"]
    carl_cats = carl_results["carl_cats"]
    carl_props = carl_results["carl_props"]

    print(f"carl_types: {carl_types}")
    print(f"carl_cats: {carl_cats}")
    print(f"carl_props: {carl_props}")

    # Process carl_results from carl_process_repository_results
    carl_results = carl_process_repository_results(carl_output_dir=carl_output_dir, debug_mode=debug_flow)
    def process_and_assign(name, df):
        df['source_path'] = df['source_path'].str.replace(r"/home/carl/(sources|output|sources_wrk|output_wrk)/", "", regex=True)
        # print(df)
        # globals()[name] = df

    # Process each DataFrame in carl_results and assign it to the global namespace
    for name in carl_results.keys():
        process_and_assign(name, carl_results[name])

    carl_object_projects = carl_results["carl_object_projects"]
    carl_object_args = carl_results["carl_object_args"]
    carl_object_positions = carl_results["carl_object_positions"]
    carl_object_callees = carl_results["carl_object_callees"]
    carl_object_parents = carl_results["carl_object_parents"]

    print(f"carl_object_projects: {carl_object_projects}")
    print(f"carl_object_args: {carl_object_args}")
    print(f"carl_object_positions: {carl_object_positions}")
    print(f"carl_object_callees: {carl_object_callees}")
    print(f"carl_object_parents: {carl_object_parents}")

    # Get names of columns for each DataFrame in carl_results
    column_names = {name: df.columns.tolist() for name, df in carl_results.items()}

    # Assume carl_object_positions is a pandas DataFrame
    if 'source_path' in carl_object_positions.columns and 'pos_source_path' in carl_object_positions.columns:
        filtered_df = carl_object_positions[carl_object_positions['source_path'] != carl_object_positions['pos_source_path']]
        my_datatable(filtered_df)

        carl_object_positions = carl_object_positions.drop(columns=['pos_source_path'])
    
    # Equivalent to switch statement in R
    if transformation_source == "cloud":
        pattern_occurrence_locations = cloud_pattern_occurrence_locations
    elif transformation_source == "green":
        pattern_occurrence_locations = green_pattern_occurrence_locations
    elif transformation_source == "path":
        pass
        # pattern_occurrence_locations = path_pattern_occurrence_locations

    # Assuming my_datatable is a function that operates on pattern_occurrence_locations
    pattern_occurrence_locations = my_datatable(pattern_occurrence_locations)
    # pattern_occurrence_locations = pattern_occurrence_locations.to_pandas()

    # Perform the inner join
    pattern_occurrence_locations = pd.merge(pattern_occurrence_locations, pattern_req_processing, how='inner')

    # Select the top rows
    pattern_occurrence_locations = pattern_occurrence_locations.head(pattern_req_processing_parallel)

    # Saving to Rdata file
    file_path = os.path.join(openai_run_trace_dir, f"pattern_occurrence_locations_{irun}.Rdata")
    pattern_occurrence_locations.to_pickle(file_path)

    # Check if pattern_occurrence_locations is empty
    if pattern_occurrence_locations.empty:
        break

    # Assuming map_pattern_occurrence_object is a function
    map_pattern_occurrence_object_results = map_pattern_occurrence_object(carl_object_positions, pattern_occurrence_locations, debug_mode=debug_flow)

    # Equivalent to %>%
    pattern_occurrence_objects = map_pattern_occurrence_object_results['pattern_occurrence_objects']
    pattern_occurrence_object_ranges = map_pattern_occurrence_object_results['pattern_occurrence_object_ranges']
    
    def prepare_code_snippets(source_dir, pattern_occurrence_objects, pattern_occurrence_object_ranges, min_line_span, max_line_span, debug_mode):
        if transformation_source == "cloud":
            return prepare_code_snippets_hl_insight(source_dir, pattern_occurrence_objects, pattern_occurrence_object_ranges, min_line_span, max_line_span, debug_mode)
        elif transformation_source == "green":
            return prepare_code_snippets_hl_insight(source_dir, pattern_occurrence_objects, pattern_occurrence_object_ranges, min_line_span, max_line_span, debug_mode)
        elif transformation_source == "path":
            return prepare_code_snippets_carl_path_insight(source_dir, pattern_occurrence_objects, pattern_occurrence_object_ranges, min_line_span, max_line_span, debug_mode)
        else:
            return prepare_code_snippets_carl_insight(source_dir, pattern_occurrence_objects, pattern_occurrence_object_ranges, min_line_span, max_line_span, debug_mode)

    code_snippet_preparation_results = prepare_code_snippets(source_dir, pattern_occurrence_objects, pattern_occurrence_object_ranges, min_line_span, max_line_span, debug_flow)

    pattern_occurrence_snippets = code_snippet_preparation_results['pattern_occurrence_snippets']
    pattern_occurrence_objects = code_snippet_preparation_results['pattern_occurrence_objects']

    pattern_occurrence_snippets = my_datatable(pattern_occurrence_snippets)
    
    build_carl_from_to_results = build_carl_from_to(carl_object_callees = carl_object_callees, carl_object_projects = carl_object_projects, debug_mode=debug_flow)

    carl_from_to = build_carl_from_to_results['carl_from_to']
    
    build_carl_exceptions_results = build_carl_exceptions(carl_from_to=carl_from_to, carl_object_projects=carl_object_projects, debug_mode=debug_flow)
    
    carl_exceptions = build_carl_exceptions_results['carl_exceptions']
    
    build_carl_signatures_results = build_carl_signatures(carl_object_args=carl_object_args, carl_object_positions=carl_object_positions, debug_mode=debug_flow)

    carl_signatures = build_carl_signatures_results["carl_signatures"]

    carl_signatures = my_datatable(carl_signatures)

    # Group by the specified columns and count occurrences
    transformation_objects = (
        pattern_occurrence_objects
        .groupby(['scan_id', 'source_path'])
        .size()
        .reset_index(name='nb_req')
        .rename(columns={'scan_id':'to', 'source_path':'to_path'})
    )

    direct_caller_census_results = direct_caller_census(carl_from_to=carl_from_to, carl_signatures=carl_signatures, transformation_objects=transformation_objects,    debug_mode=debug_flow)

    pattern_occurrence_direct_impact_stats = direct_caller_census_results['transformation_direct_impact_stats']
    pattern_occurrence_direct_impact_code = direct_caller_census_results['transformation_direct_impact_code']
    pattern_occurrence_direct_impact_code_long = direct_caller_census_results['transformation_direct_impact_code_long']
    pattern_occurrence_direct_impact = direct_caller_census_results['transformation_direct_impact']
    pattern_occurrence_isolated = direct_caller_census_results['transformation_isolated']

    # Check that variables are assigned correctly
    # print(globals().keys())

    pattern_occurrence_direct_impact_stats = my_datatable(pattern_occurrence_direct_impact_stats)
  
    pattern_occurrence_direct_impact_code = my_datatable(pattern_occurrence_direct_impact_code)

    enclosed_census_results = enclosed_census(transformation_object_ranges=pattern_occurrence_object_ranges, transformation_objects=pattern_occurrence_objects, debug_mode=debug_flow)

    pattern_occurrence_enclosed = enclosed_census_results['transformation_enclosed']
    pattern_occurrence_enclosed_long = enclosed_census_results['transformation_enclosed_long']
    pattern_occurrence_enclosed_code = enclosed_census_results['transformation_enclosed_code']
    pattern_occurrence_enclosed_code_long = enclosed_census_results['transformation_enclosed_code_long']

    # pattern_occurrence_enclosed_code = my_datatable(pattern_occurrence_enclosed_code)

    build_carl_graph_results = build_carl_graph(carl_from_to, pattern_occurrence_objects, carl_signatures, carl_object_projects, pattern_occurrence_object_ranges, debug_mode=False)

    program_flow_graph = build_carl_graph_results['program_flow_graph']
    program_flow_graph_close_focus = build_carl_graph_results['program_flow_graph_close_focus']


    # Assuming program_flow_graph_close_focus is a DataFrame and G is a graph created using networkx
    program_flow_graph_close_focus = pd.DataFrame({
        'shortname': ['A', 'B', 'C'],
        'signature': ['sig1', 'sig2', 'sig3'],
        'nb_alerts': [1, 0, 3],
        'link_type': ['defineLink', 'other', 'other'],
        'highlight': [True, False, True],
        'techno': ['tech1', 'tech2', 'tech3']
    })

    # Creating the graph
    G = nx.DiGraph()

    # Adding nodes with attributes
    for idx, row in program_flow_graph_close_focus.iterrows():
        G.add_node(row['shortname'], 
                label=row['shortname'],
                title=row['signature'],
                value=row['nb_alerts'],
                transform="YES" if row['nb_alerts'] > 0 else "NO",
                shape="box",
                color_background="white",
                borderWidth=2 if row['nb_alerts'] > 0 else 0.5,
                color_border="darkred" if row['nb_alerts'] > 0 else "darkblue",
                font_size=20 if row['nb_alerts'] > 0 else 10,
                font_color="darkred" if row['nb_alerts'] > 0 else "darkblue")

    # Adding edges with attributes
    for idx, row in program_flow_graph_close_focus.iterrows():
        if pd.notna(row['link_type']):
            G.add_edge(row['shortname'], row['shortname'], 
                    label=None if row['link_type'] == 'defineLink' else row['link_type'],
                    arrows="to",
                    value=5 if row['highlight'] else 1,
                    color="darkgrey" if row['link_type'] == 'defineLink' else "darkblue")

    # Drawing the graph
    pos = nx.spring_layout(G, seed=1234)  # layout_with_fr equivalent
    node_labels = nx.get_node_attributes(G, 'label')
    node_colors = [G.nodes[node]['color_border'] for node in G.nodes]
    node_sizes = [G.nodes[node]['font_size'] * 10 for node in G.nodes]
    edge_colors = nx.get_edge_attributes(G, 'color').values()

    nx.draw(G, pos, labels=node_labels, node_color=node_colors, node_size=node_sizes, edge_color=edge_colors, with_labels=True)

    plt.show()

    json_fix_resp = '''
    {
    "updated":"<YES/NO to state if you updated the code or not (if you believe it did not need fixing)>",
    "comment":"<explain here what you updated (or the reason why you did not update it)>",
    "missing_information":"<list here information needed to finalize the code (or NA if nothing is needed or if the code was not updated)>",
    "signature_impact":"<YES/NO/UNKNOWN, to state here if the signature of the code will be updated as a consequence of changed parameter list, types, return type, etc.>",
    "exception_impact":"<YES/NO/UNKNOWN, to state here if the exception handling related to the code will be update, as a consequence of changed exception thrown or caught, etc.>",
    "enclosed_impact":"<YES/NO/UNKNOWN, to state here if the code update could impact code enclosed in it in the same source file, such as methods defined in updated class, etc.>",
    "other_impact":"<YES/NO/UNKNOWN, to state here if the code update could impact any other code referencing this code>",
    "impact_comment":"<comment here on signature, exception, enclosed, other impacts on any other code calling this one (or NA if not applicable)>",
    "code":"<the fixed code goes here (or original code if the code was not updated)>"
    }
    '''

    openai_gen_results = gen_code_connected_json(pattern_occurrence_snippets=pattern_occurrence_snippets, pattern_occurrence_direct_impact_code=pattern_occurrence_direct_impact_code, pattern_req_instructions=pattern_req_instructions, carl_signatures=carl_signatures, carl_exceptions=carl_exceptions, json_resp=json_fix_resp, transformation_target=transformation_target, openai_snippet_trace_dir=openai_snippet_trace_dir, openai_model=openai_model, openai_api_url=openai_api_url, openai_key=openai_key, openai_invocation_delay=openai_invocation_delay, debug_mode=debug_flow)

    pattern_occurrence_snippets_processing = openai_gen_results['pattern_occurrence_snippets_processing']

    pattern_occurrence_snippets_processing = my_datatable(pattern_occurrence_snippets_processing)


    json_results = process_transform_json(pattern_occurrence_snippets_processing = pattern_occurrence_snippets_processing, json_resp = json_fix_resp, openai_json_check_trace_dir = openai_check_trace_dir, openai_model = openai_check_model, openai_api_url = openai_api_url, openai_key = openai_key, openai_invocation_delay = openai_invocation_delay, debug_mode = debug_flow)

    pattern_occurrence_snippets_json = json_results['pattern_occurrence_snippets_json']

    pattern_occurrence_snippets_json = my_datatable(pattern_occurrence_snippets_json)

    # Table equivalent in Python for each column
    error_table = pattern_occurrence_snippets_json['error'].value_counts()
    updated_table = pattern_occurrence_snippets_json['updated'].value_counts()
    signature_impact_table = pattern_occurrence_snippets_json['signature_impact'].value_counts()
    exception_impact_table = pattern_occurrence_snippets_json['exception_impact'].value_counts()
    enclosed_impact_table = pattern_occurrence_snippets_json['enclosed_impact'].value_counts()
    other_impact_table = pattern_occurrence_snippets_json['other_impact'].value_counts()

    # Print the tables
    print("Error Table:")
    print(error_table)

    print("\nUpdated Table:")
    print(updated_table)

    print("\nSignature Impact Table:")
    print(signature_impact_table)

    print("\nException Impact Table:")
    print(exception_impact_table)

    print("\nEnclosed Impact Table:")
    print(enclosed_impact_table)

    print("\nOther Impact Table:")
    print(other_impact_table)

    # Generate a skim-like profile report
    # profile = ProfileReport(pattern_occurrence_snippets_json, title="Pattern Occurrence Snippets Profile", explorative=True)

    # To display the report in a Jupyter notebook, you can use:
    # profile.to_notebook_iframe()

    # To save the report as an HTML file:
    # profile.to_file("pattern_occurrence_snippets_profile.html")

    # Create an empty DataFrame with the specified columns
    dep_check_json_empty = pd.DataFrame(columns=[
        "prompt", "response", "dep_scan_id", "dep_path", "dep_type",
        "dep_signature", "dep_fullname", "dep_name", "dep_line_start",
        "dep_line_end", "techno", "parent_info", "callee_info", "updated",
        "comment", "missing_information", "signature_impact", "exception_impact",
        "other_impact", "impact_comment", "code", "error"
    ])

    merge_results = merge_all_code_json(dep_check_json=dep_check_json_empty, pattern_occurrence_snippets_json=pattern_occurrence_snippets_json, source_dir=source_dir,gen_source_dir=gen_source_dir, comment_str="//", comment_before=True, debug_mode=debug_flow)

    all_code_merge_processing = merge_results['all_code_merge_processing']

    all_code_merge_processing = my_datatable(all_code_merge_processing)

    transformed_code_merge_processing = all_code_merge_processing.rename(columns={
        'dep_path': 'file',
        'dep_scan_id': 'scan_id',
        'dep_line_start': 'line_start',
        'dep_line_end': 'line_end'
    })[['file', 'scan_id', 'input', 'output', 'line_start', 'line_end', 'technologies', 'requirements', 'from', 'to']]

    # Create empty DataFrames for all_dep_check_preparation and all_dep_check_processing
    all_dep_check_preparation = pd.DataFrame()
    all_dep_check_processing = pd.DataFrame()

    transformation_direct_impact_code_long = pattern_occurrence_direct_impact_code_long
    transformation_enclosed_code_long = pattern_occurrence_enclosed_code_long
    code_merge_processing = transformed_code_merge_processing
    transformation_snippets_json = pattern_occurrence_snippets_json

    # Calculate the number of rows where 'updated' column is 'YES'
    nb_dep_to_process = transformation_snippets_json[transformation_snippets_json['updated'] == "YES"].shape[0]

    while nb_dep_to_process > 0:
        # START of DEP loop ####
        
        if all_dep_check_processing.shape[0] > 0:
            # Filter and join operation
            transformation_objects = all_dep_check_processing.merge(
                dep_check_json[dep_check_json['updated'] == "YES"], 
                on='dep_scan_id'
            ).groupby(['dep_scan_id', 'dep_path']).size().reset_index(name='nb_req').rename(columns={
                'dep_scan_id': 'to',
                'dep_path': 'to_path'
            })
            
            # Placeholder for the direct_caller_census function
            direct_caller_census_results = direct_caller_census(carl_from_to=carl_from_to, carl_signatures=carl_signatures,transformation_objects=transformation_objects, debug_mode=debug_flow)
            
            transformation_direct_impact_stats = direct_caller_census_results['transformation_direct_impact_stats']
            transformation_direct_impact_code = direct_caller_census_results['transformation_direct_impact_code']
            transformation_direct_impact_code_long = direct_caller_census_results['transformation_direct_impact_code_long']
            transformation_direct_impact = direct_caller_census_results['transformation_direct_impact']
            transformation_isolated = direct_caller_census_results['transformation_isolated']

            # Placeholder for my_datatable function
            transformation_direct_impact_stats = my_datatable(transformation_direct_impact_stats)
            transformation_direct_impact_code = my_datatable(transformation_direct_impact_code)
            
            # Placeholder for the get_object_range function
            transformation_object_ranges = get_object_range(carl_object_positions=carl_object_positions,object_path_list=transformation_objects['to_path'].unique(), debug_mode=debug_flow)
            
            # Placeholder for the enclosed_census function
            enclosed_census_results = enclosed_census(transformation_object_ranges=transformation_object_ranges,transformation_objects=transformation_objects.rename(columns={'to': 'scan_id', 'to_path': 'source_path'}), debug_mode=debug_flow)
            
            transformation_enclosed = enclosed_census_results['transformation_enclosed']
            transformation_enclosed_long = enclosed_census_results['transformation_enclosed_long']
            transformation_enclosed_code = enclosed_census_results['transformation_enclosed_code']
            transformation_enclosed_code_long = enclosed_census_results['transformation_enclosed_code_long']

            # Placeholder for my_datatable function
            transformation_enclosed_code = my_datatable(transformation_enclosed_code)

            # Transformation for snippets JSON
            direct_impact_df = (transformation_direct_impact_code_long[['to']].rename(columns={'to': 'dep_scan_id'}) 
                                if 'to' in transformation_direct_impact_code_long.columns else pd.DataFrame(columns=['dep_scan_id']))
            enclosed_code_df = (transformation_enclosed_code_long[['from']].rename(columns={'from': 'dep_scan_id'}) 
                                if 'from' in transformation_enclosed_code_long.columns else pd.DataFrame(columns=['dep_scan_id']))

            transformation_snippets_json = dep_check_json.merge(pd.concat([direct_impact_df, enclosed_code_df]), on='dep_scan_id', how='inner'
            ).rename(columns=lambda x: x.replace('dep_', ''))

            if 'enclosed_impact' not in transformation_snippets_json.columns:
                transformation_snippets_json['enclosed_impact'] = "NA"

        if transformation_snippets_json.shape[0] > 0:
            # Placeholder for prep_dependent_code_json function
            prep_dependent_code_check_results = prep_dependent_code_json(source_dir=source_dir, check_source_dir=check_source_dir,transformation_direct_impact_code_long=transformation_direct_impact_code_long,transformation_enclosed_code_long=transformation_enclosed_code_long, code_merge_processing=code_merge_processing,transformation_snippets_json=transformation_snippets_json, carl_signatures=carl_signatures,carl_object_positions=carl_object_positions, debug_mode=debug_flow)

            # Assign results to global environment
            dep_check_preparation = prep_dependent_code_check_results['dep_check_preparation']

            # Placeholder for my_datatable function
            dep_check_preparation = my_datatable(dep_check_preparation)

            all_dep_check_preparation = pd.concat([all_dep_check_preparation, dep_check_preparation]).drop_duplicates()

            if dep_check_preparation[(~dep_check_preparation['parent_info'].isna()) | (~dep_check_preparation['callee_info'].isna())].shape[0] > 0:
                json_dep_resp = '''
                {
                    "updated":"<YES/NO to state if you updated the dependent code or not (if you believe it did not need updating)>",
                    "comment":"<explain here what you updated (or NA if the dependent code does not need to be updated)>",
                    "missing_information":"<list here information needed to finalize the dependent code (or NA if nothing is needed or if the dependent code was not updated)>",
                    "signature_impact":"<YES/NO/UNKNOWN, to state here if the signature of the dependent code will be updated as a consequence of changed parameter list, types, return type, etc.>",
                    "exception_impact":"<YES/NO/UNKNOWN, to state here if the exception handling related to the dependent code will be update, as a consequence of changed exception thrown or caugth, etc.>",
                    "enclosed_impact":"<YES/NO/UNKNOWN, to state here if the dependent code update could impact further code enclosed in it in the same source file, such as methods defined in updated class, etc.>",
                    "other_impact":"<YES/NO/UNKNOWN, to state here if the dependent code update could impact any other code referencing this code>",
                    "impact_comment":"<comment here on signature, exception, enclosed, other impacts on any other code calling this one (or NA if not applicable)>",
                    "code":"<the updated dependent code goes here (or original dependent code if the dependent code was not updated)>"
                }
                '''

                # Placeholder for check_dependent_code_json function
                check_dependent_code_check_results = check_dependent_code_json(
                    source_dir=source_dir,
                    check_source_dir=check_source_dir,
                    dep_check_preparation=dep_check_preparation,
                    carl_signatures=carl_signatures,
                    carl_object_positions=carl_object_positions,
                    json_resp=json_dep_resp,
                    openai_dependent_check_trace_dir=openai_dependent_check_trace_dir,
                    openai_model=openai_model,
                    openai_api_url=openai_api_url,
                    openai_key=openai_key,
                    openai_invocation_delay=openai_invocation_delay,
                    debug_mode=debug_flow
                )

                dep_check_processing = check_dependent_code_check_results['dep_check_processing']

                # Placeholder for my_datatable function
                dep_check_processing = my_datatable(dep_check_processing)

                all_dep_check_processing = pd.concat([all_dep_check_processing, dep_check_processing]).drop_duplicates()
                print(all_dep_check_processing)

                if dep_check_processing.shape[0] == 0:
                    # Use pdb for debugging
                    import pdb; pdb.set_trace()

                if 'prompt' not in dep_check_processing.columns:
                    # Use pdb for debugging
                    import pdb; pdb.set_trace()

                # Placeholder for process_dep_check_json function
                json_results = process_dep_check_json(dep_check_processing=dep_check_processing, json_resp=json_dep_resp,openai_json_check_trace_dir=openai_check_trace_dir, openai_model=openai_check_model, openai_api_url=openai_api_url,openai_key=openai_key, openai_invocation_delay=openai_invocation_delay, debug_mode=debug_flow)

                # Assign results to global environment
                dep_check_json = json_results['dep_check_json']

                # Placeholder for my_datatable function
                dep_check_json = my_datatable(dep_check_json)

                print(dep_check_json['error'].value_counts())
                print(dep_check_json['updated'].value_counts())
                print(dep_check_json['missing_information'].value_counts())
                print(dep_check_json['signature_impact'].value_counts())
                print(dep_check_json['exception_impact'].value_counts())
                print(dep_check_json['other_impact'].value_counts())

                # Assuming skim function is defined elsewhere
                dep_check_json.skim()

                nb_dep_to_process = dep_check_json[dep_check_json['updated'] == "YES"].shape[0]

            else:
                nb_dep_to_process = 0

        else:
            nb_dep_to_process = 0

    # Check if the DataFrame all_dep_check_processing is empty
    if all_dep_check_processing.shape[0] == 0:
        # Use pdb for debugging
        import pdb; pdb.set_trace()

    # Check if 'prompt' column exists in the DataFrame all_dep_check_processing
    if 'prompt' not in all_dep_check_processing.columns:
        # Use pdb for debugging
        import pdb; pdb.set_trace()

    print("================\n\n", "DEP loop done")
    # Use pdb for debugging
    import pdb; pdb.set_trace()

    # Placeholder for process_dep_check_json function
    json_results = process_dep_check_json(dep_check_processing=all_dep_check_processing, json_resp=json_dep_resp,openai_json_check_trace_dir=openai_check_trace_dir, openai_model=openai_check_model, openai_api_url=openai_api_url, openai_key=openai_key,openai_invocation_delay=openai_invocation_delay, debug_mode=debug_flow)

    # Assign results to global environment
    dep_check_json = json_results['dep_check_json']

    # Placeholder for my_datatable function
    dep_check_json = my_datatable(dep_check_json)


    # Placeholder for merge_all_code_json function
    merge_results = merge_all_code_json(dep_check_json=dep_check_json, pattern_occurrence_snippets_json=pattern_occurrence_snippets_json,source_dir=source_dir, gen_source_dir=gen_source_dir, comment_str="//", comment_before=True, debug_mode=debug_flow)

    # Assign results to global environment
    all_code_merge_processing = merge_results['all_code_merge_processing']

    # Placeholder for my_datatable function
    all_code_merge_processing = my_datatable(all_code_merge_processing)

    # Placeholder for build_new_input function
    new_input_results = build_new_input(check_source_dir=check_source_dir, gen_source_dir=gen_source_dir, source_dir=source_dir,debug_mode=debug_flow)

    # Assign results to global environment
    cg_file_moving = new_input_results['cg_file_moving']


    # Start the rescan
    print(time.ctime())
    hl_scan_it(source_dir, hl_output_dir)
    print(time.ctime())

    # Remove the carl_output_dir directory
    shutil.rmtree(carl_output_dir, ignore_errors=True)

    # Check if the carl_output_dir still exists
    if os.path.exists(carl_output_dir):
        import pdb; pdb.set_trace()  # Start the debugger

    # Run the carl_scan_it function
    carl_scan_it(target_application, source_dir, carl_output_dir, carl_cfg_dir, carl_ext_dir)
    print(time.ctime())

    # Debug loop
    debug_loop = False
    if debug_loop:
        import pdb; pdb.set_trace()  # Start the debugger

    print("===============================\n\nFIX loop done")