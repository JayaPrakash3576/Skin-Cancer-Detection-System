def map_severity(label):
    if label == 'Malignant':
        return 'High'
    elif label == 'Benign':
        return 'Low'
    else:
        return 'Moderate'  # fallback, though not used in binary case
