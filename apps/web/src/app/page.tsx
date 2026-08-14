import { ExperimentLab } from "@/components/experiment-lab";
import { ShaderField } from "@/components/shader-field";

export default function Home() {
  return (
    <>
      <header className="site-header">
        <a href="#top" className="wordmark">JUMP <span>RESEARCH</span></a>
        <p>SIMULATION PREVIEW · DECLARATIVE ENGINE · V2</p>
      </header>
      <section id="top" className="hero">
        <ShaderField />
        <div className="hero-copy">
          <p className="hero-kicker">A SMALL WORLD. ONE CHANGED RULE.</p>
          <h1>Run the<br /><em>counterfactual.</em></h1>
          <p>Describe a visual counterfactual. Inspect the plan. Compare the worlds.</p>
          <a href="#experiment">Begin experiment <span>↘</span></a>
        </div>
        <div className="hero-note"><span>MODE</span><strong>SIMULATION PREVIEW</strong><span>ENGINE</span><strong>DETERMINISTIC / 2D</strong><span>CLAIMS</span><strong>MODEL-BOUND ONLY</strong></div>
      </section>
      <div id="experiment"><ExperimentLab /></div>
      <footer><p>JUMP RESEARCH · DECLARATIVE SIMULATION PREVIEW</p><p>Frames are simulator states, not observations or learned reconstructions.</p></footer>
    </>
  );
}
