import yaml
import jsonschema

# schema laden
with open("C:\\Users\\roord\\Documents\\Stage_git\\test_pipeline\\schema_pipeline.yaml") as f:
    schema = yaml.safe_load(f)

# yaml pipeline laden
with open("C:\\Users\\roord\\Documents\\Stage_git\\test_pipeline\\pipeline_test.yaml") as f:
    data = yaml.safe_load(f)

# valideren
jsonschema.validate(instance=data, schema=schema)

print("YAML is geldig volgens het schema!")