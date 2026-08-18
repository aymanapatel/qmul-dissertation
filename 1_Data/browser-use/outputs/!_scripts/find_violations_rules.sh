#!/bin/bash

# Default values
Rule="input-image-alt"
AxeCoreDir="../axe-core"

# 1. Parse command-line arguments for the --rule flag
while [[ "$#" -gt 0 ]]; do
  case $1 in
    --rule)
      Rule="$2"
      shift 2
      ;;
    *)
      echo "Unknown parameter passed: $1"
      exit 1
      ;;
  esac
done

# 2. Execute find and pass the Rule dynamically
find "$AxeCoreDir" -name "*.json" -exec sh -c '
  # Extract the rule passed as the first argument to the subshell
  target_rule="$1"
  shift 
  
  for file; do
    # Use --arg to safely pass the shell variable into jq
    if jq -e --arg r "$target_rule" "any(.. | objects | .violations? | arrays | .[]; .id? == \$r)" "$file" >/dev/null 2>&1; then
      echo "Found in file path: $file"
    fi
  done
' _ "$Rule" {} +