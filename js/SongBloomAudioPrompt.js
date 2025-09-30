// SongBloom Audio Prompt File Selector
app.registerExtension({
    name: "SongBloom.AudioPrompt",
    async beforeRegisterNodeDef(nodeType, nodeData, _app) {
        if (nodeData.name === "SongBloomAudioPrompt") {
            const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                originalOnNodeCreated?.apply(this, arguments);
                
                // Add file picker button after the node is created
                this.addWidget("button", "📁 选择音频文件", () => {
                    this.pickAudioFile();
                }, {
                    serialize: false
                });
            };

            // Add file picker method
            nodeType.prototype.pickAudioFile = function() {
                // Create file input element
                const input = document.createElement('input');
                input.type = 'file';
                input.accept = '.wav,.flac,.mp3,.m4a,.ogg,.aac,.wma';
                input.style.display = 'none';
                
                input.onchange = (event) => {
                    const file = event.target.files[0];
                    if (file) {
                        // Update the audio_file widget value
                        const audioFileWidget = this.widgets.find(w => w.name === 'audio_file');
                        if (audioFileWidget) {
                            // Use file path if available, otherwise use file name
                            audioFileWidget.value = file.path || file.name;
                            this.setDirtyCanvas(true, true);
                        }
                    }
                };
                
                // Trigger file picker
                document.body.appendChild(input);
                input.click();
                document.body.removeChild(input);
            };
        }
    }
});
