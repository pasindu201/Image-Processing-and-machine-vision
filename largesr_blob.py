largest_blob_idx = np.argmax(radii_all)

# Get the coordinates and radius of the largest blob
largest_blob_x = x_all[largest_blob_idx]
largest_blob_y = y_all[largest_blob_idx]
largest_blob_radius = radii_all[largest_blob_idx]

# Create a copy of the input image
output_img_with_largest_blob = img.copy()

# Draw the largest blob on the image
cv.circle(output_img_with_largest_blob, (largest_blob_y, largest_blob_x), int(largest_blob_radius), (0, 0, 255), 2)

# Display the image with the largest blob
plt.figure(figsize=(8, 8))
plt.imshow(cv.cvtColor(output_img_with_largest_blob, cv.COLOR_BGR2RGB))
plt.axis('off')
plt.title('Largest Blob')
plt.show()
print("radius :",largest_blob_radius)
print("sigma :",largest_blob_radius/np.sqrt(2))