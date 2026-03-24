import base64
import zlib
import urllib.request
import os
import re

def generate_diagram():
    mermaid_code = """
%%{init: {'theme': 'base', 'themeVariables': { 'primaryTextColor': '#000000', 'lineColor': '#000000', 'clusterBkg': '#fafafa', 'clusterBorder': '#333333'}}}%%
graph BT
    classDef pink fill:#ffcdd2,stroke:#333,stroke-width:1px,color:#000;
    classDef blue fill:#b3e5fc,stroke:#333,stroke-width:1px,color:#000;
    classDef yellow fill:#fff9c4,stroke:#333,stroke-width:1px,color:#000;
    classDef orange fill:#ffe0b2,stroke:#333,stroke-width:1px,color:#000;
    classDef add fill:#fff9c4,stroke:#333,stroke-width:1px,color:#000,shape:circle;

    Input[Input Tokens]:::pink --> WTE[Word Token Embedding]:::pink
    
    WTE --> B_In[Input]:::blue

    subgraph Block [N x Transformer Blocks]
        direction BT
        B_In --> LN1[RMSNorm]:::yellow
        LN1 --> Attn[CausalSelfAttention<br>GQA + RoPE + Conv1d]:::orange
        Attn --> Add1((Add)):::add
        B_In --> Add1

        Add1 --> LN2[RMSNorm]:::yellow
        LN2 --> MoE[MoE Layer<br>8 Experts, Top-2]:::blue
        MoE --> Add2((Add)):::add
        Add1 --> Add2
        Add2 --> B_Out[Output]:::blue
    end

    B_Out --> LNF[RMSNorm]:::yellow
    LNF --> LM[LM Head Linear]:::blue
    LM --> Output[Logits]:::pink
"""

    print("Generating SVG diagram using Kroki API...")
    
    # Kroki encoding: zlib compress then base64url encode
    compressed = zlib.compress(mermaid_code.encode('utf-8'), 9)
    encoded = base64.urlsafe_b64encode(compressed).decode('ascii')
    
    url = f"https://kroki.io/mermaid/svg/{encoded}"
    
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        with urllib.request.urlopen(req) as response:
            svg_data = response.read().decode('utf-8')
            
            # Inject a white background rectangle explicitly so it's guaranteed visible in dark mode
            if "<svg" in svg_data:
                # Find the end of the opening <svg ...> tag
                svg_tag_end = svg_data.find(">", svg_data.find("<svg")) + 1
                bg_rect = '\n<rect width="100%" height="100%" fill="white"/>\n'
                svg_data = svg_data[:svg_tag_end] + bg_rect + svg_data[svg_tag_end:]
            
            with open('architecture.svg', 'w', encoding='utf-8') as f:
                f.write(svg_data)
                
        print("Successfully saved architecture.svg")
        
        # Clean up the old png if it exists
        if os.path.exists('architecture.png'):
            os.remove('architecture.png')
            
    except Exception as e:
        print(f"Failed to generate diagram: {e}")

if __name__ == "__main__":
    generate_diagram()
