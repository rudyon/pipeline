import base64
import zlib
import urllib.request
import os

def generate_diagram():
    mermaid_code = """
graph TD
    classDef default fill:#f9f9f9,stroke:#333,stroke-width:2px;
    classDef block fill:#e1f5fe,stroke:#01579b,stroke-width:2px;
    classDef emb fill:#fff3e0,stroke:#e65100,stroke-width:2px;
    classDef norm fill:#f3e5f5,stroke:#4a148c,stroke-width:2px;
    classDef module fill:#e8f5e9,stroke:#1b5e20,stroke-width:2px;
    classDef add fill:#eeeeee,stroke:#616161,stroke-width:2px,shape:circle;

    Input[Input Tokens]:::default --> WTE[Word Token Embedding]:::emb
    WTE --> BlockLoop[N x Transformer Blocks]:::block
    
    subgraph Transformer Block [Transformer Block]
        direction TB
        B_In[Input]:::default --> LN1[RMSNorm]:::norm
        LN1 --> Attn[CausalSelfAttention<br>GQA + RoPE + Depthwise Conv1d]:::module
        Attn --> Add1((+)):::add
        B_In --> Add1
        
        Add1 --> LN2[RMSNorm]:::norm
        LN2 --> MoE[MoE Layer<br>8 Experts, Top-2 Routing]:::module
        
        MoE --> Add2((+)):::add
        Add1 --> Add2
        Add2 --> B_Out[Output]:::default
    end
    
    BlockLoop -.-> B_In
    B_Out -.-> LNF[RMSNorm]:::norm
    LNF --> LM[LM Head Linear]:::emb
    LM --> Output[Logits]:::default
"""

    print("Generating diagram using Kroki API...")
    
    # Kroki encoding: zlib compress then base64url encode
    compressed = zlib.compress(mermaid_code.encode('utf-8'), 9)
    encoded = base64.urlsafe_b64encode(compressed).decode('ascii')
    
    url = f"https://kroki.io/mermaid/png/{encoded}"
    
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        with urllib.request.urlopen(req) as response:
            with open('architecture.png', 'wb') as f:
                f.write(response.read())
        print("Successfully saved architecture.png")
    except Exception as e:
        print(f"Failed to generate diagram: {e}")

if __name__ == "__main__":
    generate_diagram()
