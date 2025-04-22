/**
 * ASIC - AI for Satellite Image Classification
 * Main JavaScript file for handling image upload, processing, and display
 */

// Get DOM elements
const dropZone = document.querySelector('.drop-zone');          // The drag and drop area
const fileInput = document.getElementById('image');            // Hidden file input
const uploadForm = document.getElementById('uploadForm');      // The upload form
const originalImage = document.getElementById('originalImage'); // Container for original image
const segmentedImage = document.getElementById('segmentedImage'); // Container for segmented image
const consoleOutput = document.getElementById('consoleOutput'); // Console output area
const processBtn = document.querySelector('.process-btn');     // Process image button

/**
 * Resets the form to its initial state
 * - Clears file input
 * - Hides both images
 * - Disables process button
 * - Resets console message
 */
function resetForm() {
    fileInput.value = '';
    originalImage.style.display = 'none';
    segmentedImage.style.display = 'none';
    processBtn.disabled = true;
    consoleOutput.innerText = '> Ready to process image... Select and image and press "Process Image" button.';
}

// Initialize form state when page loads
resetForm();

/**
 * Drag and Drop Event Handlers
 * Prevents default browser behavior for drag and drop events
 */
['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
    dropZone.addEventListener(eventName, preventDefaults, false);
});

function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
}

/**
 * Visual Feedback for Drag and Drop
 * Adds/removes highlight class when dragging files over drop zone
 */
['dragenter', 'dragover'].forEach(eventName => {
    dropZone.addEventListener(eventName, highlight, false);
});

['dragleave', 'drop'].forEach(eventName => {
    dropZone.addEventListener(eventName, unhighlight, false);
});

function highlight(e) {
    dropZone.classList.add('highlight');
}

function unhighlight(e) {
    dropZone.classList.remove('highlight');
}

/**
 * Handle File Drop
 * Processes files when dropped onto the drop zone
 */
dropZone.addEventListener('drop', handleDrop, false);

function handleDrop(e) {
    const dt = e.dataTransfer;
    const files = dt.files;
    handleFiles(files);
}

/**
 * Handle File Input Change
 * Processes files when selected through the file input
 */
fileInput.addEventListener('change', function() {
    handleFiles(this.files);
});

/**
 * Process Selected Files
 * @param {FileList} files - List of files to process
 * Handles file validation and preview display
 */
function handleFiles(files) {
    if (files.length > 0) {
        const file = files[0];
        if (file.type.startsWith('image/')) {
            // Update file input with selected file
            const dataTransfer = new DataTransfer();
            dataTransfer.items.add(file);
            fileInput.files = dataTransfer.files;

            // Display image preview
            const imageURL = URL.createObjectURL(file);
            originalImage.src = imageURL;
            originalImage.style.display = 'block';
            segmentedImage.style.display = 'none';
            
            // Enable process button
            processBtn.disabled = false;
            
            // Update console with selection message
            consoleOutput.innerText = `> Image '${file.name}' selected. Ready to process...`;
        } else {
            consoleOutput.innerText = `> Error: Please select an image file.`;
            processBtn.disabled = true;
        }
    }
}

/**
 * Form Submission Handler
 * Handles the image upload and processing
 */
uploadForm.addEventListener('submit', async function(e) {
    e.preventDefault();
    
    // Validate that an image is selected
    if (!fileInput.files[0]) {
        consoleOutput.innerText = `> Error: Please select an image first.`;
        return;
    }

    const file = fileInput.files[0];
    const formData = new FormData();
    formData.append('image', file);

    // Update UI for processing state
    processBtn.disabled = true;
    processBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Processing...';
    consoleOutput.innerText += `\n> Uploading '${file.name}'...`;

    try {
        // Send image to server for processing
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });

        const result = await response.json();
        consoleOutput.innerText += `\n> ${result.message}`;
        
        // Display segmented image if processing was successful
        if (result.segmented_image) {
            segmentedImage.src = result.segmented_image;
            segmentedImage.style.display = 'block';
            consoleOutput.innerText += `\n> Done! Segmented output is now displayed.`;
            consoleOutput.innerText += `    ` + result.analysis;
        }
    } catch (error) {
        consoleOutput.innerText += `\n> Error: ${error.message}`;
    } finally {
        // Reset button state regardless of success or failure
        processBtn.innerHTML = '<i class="fas fa-magic"></i> Process Image';
        processBtn.disabled = false;
    }
});

/**
 * Make Drop Zone Clickable
 * Triggers file input when drop zone is clicked
 */
dropZone.addEventListener('click', () => {
    fileInput.click();
});
