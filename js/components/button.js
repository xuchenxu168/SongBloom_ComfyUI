import { BaseUIComponent } from "./base.js";

export class ButtonComponent extends BaseUIComponent {
    constructor(parent, options = {}) {
        super(parent, options);
        this.text = options.text || "";
        this.onClick = options.onClick || null;
        this.iconPath = options.iconPath || null;
        this.iconSize = options.iconSize || 16;
        this.iconPosition = options.iconPosition || "left";
        this.color = options.color || "rgb(100, 100, 100)";
        this.hoverColor = options.hoverColor || "rgb(120, 120, 120)";
        this.pressedColor = options.pressedColor || "rgb(80, 80, 80)";
        this.textColor = options.textColor || "white";
        this.isPressed = false;
        this.isHovered = false;
        this._iconImage = null;
        
        if (this.iconPath) {
            this._loadIcon();
        }
    }

    _loadIcon() {
        if (!this.iconPath) return;
        
        this._iconImage = new Image();
        this._iconImage.onerror = (e) => {
            console.warn(`Failed to load button icon: ${this.iconPath}`, e);
            this._iconImage = null;
        };
        this._iconImage.onload = () => {
            if (this.node && this.node.setDirtyCanvas) {
                this.node.setDirtyCanvas(true, true);
            }
        };
        this._iconImage.src = this.iconPath;
    }

    setIcon(iconPath) {
        this.iconPath = iconPath;
        this._iconImage = null;
        if (iconPath) {
            this._loadIcon();
        }
    }

    _drawSelf(ctx) {
        const x = this.abs_x;
        const y = this.abs_y;
        const width = this.width;
        const height = this.height;

        // Determine button color based on state
        let buttonColor = this.color;
        if (this.isPressed) {
            buttonColor = this.pressedColor;
        } else if (this.isHovered) {
            buttonColor = this.hoverColor;
        }

        // Draw button background
        ctx.fillStyle = buttonColor;
        ctx.beginPath();
        ctx.roundRect(x, y, width, height, 4);
        ctx.fill();

        // Draw border
        ctx.strokeStyle = "rgba(255, 255, 255, 0.2)";
        ctx.lineWidth = 1;
        ctx.stroke();

        // Draw icon or text
        if (this._iconImage && this._iconImage.complete) {
            this._drawIcon(ctx, x, y, width, height);
        } else if (this.text) {
            this._drawText(ctx, x, y, width, height);
        }
    }

    _drawIcon(ctx, x, y, width, height) {
        const iconX = x + (width - this.iconSize) / 2;
        const iconY = y + (height - this.iconSize) / 2;
        
        try {
            ctx.drawImage(this._iconImage, iconX, iconY, this.iconSize, this.iconSize);
        } catch (e) {
            console.warn("Failed to draw icon:", e);
        }
    }

    _drawText(ctx, x, y, width, height) {
        ctx.fillStyle = this.textColor;
        ctx.font = "12px Arial";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(this.text, x + width / 2, y + height / 2);
    }

    _onMouseDownSelf(event, nodeRelMouse, canvas) {
        const compRelMouse = this.getNodeRelMouseToCompRelMouse(nodeRelMouse);

        if (compRelMouse.x >= 0 && compRelMouse.x <= this.width && 
            compRelMouse.y >= 0 && compRelMouse.y <= this.height) {
            
            console.log("🎵 Button clicked:", this.text || "icon button");
            
            this.isPressed = true;
            if (this.node && this.node.setDirtyCanvas) {
                this.node.setDirtyCanvas(true, true);
            }
            
            // Simple click handling - call onClick immediately
            if (this.onClick) {
                try {
                    this.onClick();
                } catch (error) {
                    console.error("🎵 Button onClick error:", error);
                }
            }
            
            // Reset pressed state after a short delay
            setTimeout(() => {
                this.isPressed = false;
                if (this.node && this.node.setDirtyCanvas) {
                    this.node.setDirtyCanvas(true, true);
                }
            }, 100);
            
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