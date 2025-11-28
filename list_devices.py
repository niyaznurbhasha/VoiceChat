import sounddevice as sd

print(sd.query_devices())
print("\nDefault device indices (input, output):", sd.default.device)
