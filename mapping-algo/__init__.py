"""
MapScore Matching Algorithm — Layer 2 (World Model) + Layer 3.1 (Coarse Filtering)

Core formula: M(u, v, T) = σ(u,v,T) · (w_c · S_cap(v,u,T) + w_n · S_need(v,T))
where w_c + w_n = 1, σ ∈ {0,1}
"""
