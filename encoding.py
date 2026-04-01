text = "hello"

# Encode the string into bytes using UTF-8 encoding
encoded_text = text.encode('utf-8')
print("Encoded text:", encoded_text)  # Output: b'hello'

# Decode the bytes back into a string
decoded_text = encoded_text.decode('utf-8')
print("Decoded text:", decoded_text)  # Output: hello

# convert to list of numbers (ASCII values)
char_list = list(encoded_text)
print("List of characters:", char_list)  # Output: [104, 101, 108, 108, 111]

# convert to list of numbers (ASCII values)
ascii_values = [ord(char) for char in decoded_text]
print("ASCII values:", ascii_values)  # Output: [104, 101, 108, 108, 111]