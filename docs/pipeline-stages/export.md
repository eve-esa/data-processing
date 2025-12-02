# Export Stage

The final stage of the pipeline where the processed documents are stored in the format required by the user. The default format is **.md**

```yaml
- name: export
  config:
    destination: "./output"
```

### destination
- **Type**: String
- **Default**: `./output`
- **Description**: Directory to save the processed files