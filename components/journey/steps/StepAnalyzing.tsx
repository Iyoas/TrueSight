import React from "react";
import "./StepAnalyzing.module.css";
import { GradientText } from "@/components/ui/shadcn-io/gradient-text";

const StepAnalyzing: React.FC = () => {
  return (
    <div className="journey-step journey-step-analyzing">
      <div className="journey-agent-bubble">
        <p>I&apos;m analyzing the image… here&apos;s what I&apos;m looking for.</p>
      </div>

      <div className="journey-analysis-list">
        <div className="journey-analysis-item">
          <div className="journey-analysis-check">
            <span className="journey-analysis-check-icon">✓</span>
          </div>
          <div className="journey-analysis-textline">
            <GradientText
              text="checking lightning conditions..."
              gradient="linear-gradient(90deg, #d1d5db 0%, #9ca3af 50%, #4b5563 100%)"
            />
          </div>
        </div>

        <div className="journey-analysis-item">
          <div className="journey-analysis-check">
            <span className="journey-analysis-check-icon">✓</span>
          </div>
          <div className="journey-analysis-textline">
            <GradientText
              text="Inspecting texture and skin details..."
              gradient="linear-gradient(90deg, #d1d5db 0%, #9ca3af 50%, #4b5563 100%)"
            />
          </div>
        </div>

        <div className="journey-analysis-item">
          <div className="journey-analysis-check">
            <span className="journey-analysis-check-icon">✓</span>
          </div>
          <div className="journey-analysis-textline">
            <GradientText
              text="Scanning for background consistency"
              gradient="linear-gradient(90deg, #d1d5db 0%, #9ca3af 50%, #4b5563 100%)"
            />
          </div>
        </div>

        <div className="journey-analysis-item journey-analysis-item--active">
          <div className="journey-analysis-check">
            <span className="journey-analysis-check-icon">✓</span>
          </div>
          <div className="journey-analysis-textline">
            <GradientText
              text="Analyzing textures and geometry"
              gradient="linear-gradient(90deg, #d1d5db 0%, #9ca3af 50%, #4b5563 100%)"
            />
          </div>
        </div>
      </div>
    </div>
  );
};

export default StepAnalyzing;
