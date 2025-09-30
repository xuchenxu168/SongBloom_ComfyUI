import { BaseUIComponent } from "./base.js";

export class CheckboxComponent extends BaseUIComponent {
    constructor(parent, options = {}) {
        super(parent, options);
        this.text = options.text || "";
        this.checked = options.checked || false;
        this.onChange = options.onChange || null;
        this.textColor = options.textColor || "white";
        this.checkboxSize = options.checkboxSize || 16;
        this.textOffset = options.textOffset || 25;
        this.isHovered = false;
    }

    setChecked(checked) {
        this.checked = checked;
        if (this.node && this.node.setDirtyCanvas) {
            this.node.setDirtyCanvas(true, true);
        }
    }

    _drawSelf(ctx) {
        const x = this.abs_x;
        const y = this.abs_y;
        const size = this.checkboxSize;

        // Draw checkbox background
        ctx.fillStyle = this.isHovered ? "rgba(100, 100, 100, 0.8)" : "rgba(80, 80, 80, 0.8)";
        ctx.beginPath();
        ctx.roundRect(x, y, size, size, 3);
        ctx.fill();

        // Draw checkbox border
        ctx.strokeStyle = "rgba(255, 255, 255, 0.3)";
        ctx.lineWidth = 1;
        ctx.stroke();

        // Draw checkmark if checked
        if (this.checked) {
            ctx.strokeStyle = "rgb(122, 156, 0)";
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.moveTo(x + 3, y + size / 2);
            ctx.lineTo(x + size / 2, y + size - 3);
            ctx.lineTo(x + size - 3, y + 3);
            ctx.stroke();
        }

        // Draw text
        if (this.text) {
            ctx.fillStyle = this.textColor;
            ctx.font = "12px Arial";
            ctx.textAlign = "left";
            ctx.textBaseline = "middle";
            ctx.fillText(this.text, x + this.textOffset, y + size / 2);
        }
    }

    _onMouseDownSelf(event, nodeRelMouse, canvas) {
        const compRelMouse = this.getNodeRelMouseToCompRelMouse(nodeRelMouse);

        if (compRelMouse.x >= 0 && compRelMouse.x <= this.width && 
            compRelMouse.y >= 0 && compRelMouse.y <= this.height) {
            
            console.log("🎵 Checkbox clicked:", this.text);
            
            // Toggle checked state
            this.checked = !this.checked;
            
            // Call onChange callback
            if (this.onChange) {
                try {
                    this.onChange(this.checked);
                } catch (error) {
                    console.error("🎵 Checkbox onChange error:", error);
                }
            }
            
            // Redraw
            if (this.node && this.node.setDirtyCanvas) {
                this.node.setDirtyCanvas(true, true);
            }
            
            return true;
        }
        return false;
    }

    _onMouseMoveSelf(event, nodeRelMouse, canvas) {
        const compRelMouse = this.getNodeRelMouseToCompRelMouse(nodeRelMouse);
        const isOver = (compRelMouse.x >= 0 && compRelMouse.x <= this.width && 
                       compRelMouse.y >= 0 && compRelMouse.y <= this.height);
        
        if (isOver !== this.isHovered) {
            this.isHovered = isOver;
            if (this.node && this.node.setDirtyCanvas) {
                this.node.setDirtyCanvas(true, true);
            }
        }
        
        return isOver;
    }

    _onMouseUpSelf(event, nodeRelMouse, canvas) {
        return false;
    }
}
