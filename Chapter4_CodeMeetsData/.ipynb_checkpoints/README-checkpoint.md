# Login to Azure
```console
az login
```

# Configure defaults
```console
az account set -s <SUBSCRIPTION_ID>
az configure --defaults group=<RESOURCE_GROUP> workspace=<ML_WORKSPACE>
```

# Create data asset
```console
az ml data create --file registerdataset.yml
```
