# Morpheus Models Datasets

Small datasets for testing the training, inference, and pipelines.

## Digital Fingerprinting (DFP) Data

### DFP Azure Logs

#### Sample Training Data
- [training-data/azure/azure-ad-logs-sample-training-data.json](./training-data/azure/azure-ad-logs-sample-training-data.json)

This is a synthetic dataset of Azure AD logs with activities of 20 accounts (85 applications involved, 3567 records in total). The activities are split to a train and an inference set. An anomaly is included in the inference set for model validation.
* The data was generated using the python [faker](https://faker.readthedocs.io/en/master/#) package. If there is any resemblance to real individuals, it is purely coincidental.

- 3239 records in total
- Time range: 2022/08/01 - 2022/08/29
- Users' log distribution:
    - 5 high volume (>= 300) users
    - 15 medium volume (~100) users
    - 5 light volume (~10) users


### DFP Cloudtrail Logs

This is a synthetic dataset of AWS CloudTrail logs events with activities from 2 entities/users in separate files. 

Files for `user-123` include a single csv and split json versions of the same data:
#### Sample Training Data
- [dfp-cloudtrail-user123-training-data.csv](./training-data/dfp-cloudtrail-user123-training-data.csv)
- [hammah-user123-training-part2.json](./training-data/cloudtrail/hammah-user123-training-part2.json)
- [hammah-user123-training-part3.json](./training-data/cloudtrail/hammah-user123-training-part3.json)
- [training-data/cloudtrail/hammah-user123-training-part4.json](./training-data/cloudtrail/hammah-user123-training-part4.json)

#### Pipeline Validation Data
- [dfp-cloudtrail-user123-validation-data-input.csv](./validation-data/dfp-cloudtrail-user123-validation-data-input.csv)
- [dfp-cloudtrail-user123-validation-data-output.csv](./validation-data/dfp-cloudtrail-user123-validation-data-input.csv)

Files for `role-g` include a single csv and split json version of the same data:
#### Sample Training Data
- [dfp-cloudtrail-role-g-training-data.csv](./training-data/dfp-cloudtrail-role-g-training-data.csv)
- [hammah-role-g-training-part1.json](./training-data/cloudtrail/hammah-role-g-training-part1.json)
- [hammah-role-g-training-part2.json](./training-data/cloudtrail/hammah-role-g-training-part1.json)

#### Pipeline Validation Data
- [dfp-cloudtrail-role-g-validation-data-input.csv](.validation-data/dfp-cloudtrail-role-g-validation-data-input.csv)
- [dfp-cloudtrail-role-g-validation-data-output.csv](.validation-data/dfp-cloudtrail-role-g-validation-data-output.csv)

## Anomalous Behavioral Profiling (ABP)

This a labeled dataset of nv-smi logs generated from a single V100 in our lab environment running either GPU malware or bengin workflows.

### Sample Training Data

- [abp-sample-nvsmi-training-data.json](./training-data/abp-sample-nvsmi-training-data.json)

### Pipeline Validation Data
Same data in both csv and jsonlines

- [abp-validation-data.csv](./validation-data/abp-validation-data.csv)
- [abp-validation-data.jsonlines](./validation-data/abp-validation-data.jsonlines)




