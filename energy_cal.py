from dataclasses import dataclass, asdict

# 能耗参数（pJ）
ACC_pJ   = 0.05448
MAC_pJ   = 4.6
SHIFT_pJ = 0.024  # 仅用于 approx_softplus 的 shifting（其余 shifting 仍按 MAC 计）

# Data movement 和 Weight access 能耗 (pJ/bit)
E_MOVE_PER_BIT = 0.18        # data movement 能耗
E_WEIGHT_PER_BIT = 0.098454  # weight access 能耗

# Bit widths
WEIGHT_BITS_FIRST_LINEAR = 32  # 第一层 linear 的 weight 是 32 bit
WEIGHT_BITS_OTHER = 8          # 其他层是 8 bit
ACTIVATION_BITS = 1            # activation/spike 是 1 bit
STATE_BITS = 32                # h, A, D 是 32 bit
ACTIVATION_BITS_in=32 #kaiwen
@dataclass
class Params:
    B: int = 1       # batch size
    L: int = 168     # window length
    D_H: int = 64    # d_inner
    D_S: int = 32    # n (state)
    T: int = 16      # bit slices
    s: float = 1.0   # 稀疏因子
    D_R: int = 21    # dt_rank
    D_R_plus_2DS: int = None  # 若为 None, 自动计算 D_R + 2*D_S
    D_IN: int = 321  # input dimension
    D_OUT_PROJ: int = 64  # in_proj output dimension

    def __post_init__(self):
        if self.D_R_plus_2DS is None:
            self.D_R_plus_2DS = self.D_R + 2 * self.D_S

def compute_counts(p: Params):
    B, L, DH, DS, T, s, DR, DR2 = p.B, p.L, p.D_H, p.D_S, p.T, p.s, p.D_R, p.D_R_plus_2DS
    D_IN = p.D_IN
    D_OUT_PROJ = p.D_OUT_PROJ
    rows = []

    def add(stage, mac=0.0, acc=0.0, shift=0.0, move_bits=0.0, weight_bits=0.0):
        """
        move_bits: data movement 的总 bit 数
        weight_bits: weight access 的总 bit 数
        """
        rows.append({
            "Stage": stage,
            "MAC": float(mac),
            "ACC": float(acc),
            "SHIFT": float(shift),
            "MOVE_BITS": float(move_bits),
            "WEIGHT_BITS": float(weight_bits),
            "MAC_energy_pJ": float(mac) * MAC_pJ,
            "ACC_energy_pJ": float(acc) * ACC_pJ,
            "SHIFT_energy_pJ": float(shift) * SHIFT_pJ,
            "MOVE_energy_pJ": float(move_bits) * E_MOVE_PER_BIT,
            "WEIGHT_energy_pJ": float(weight_bits) * E_WEIGHT_PER_BIT,
        })

    # =============================================================
    # 你的汇总逐项实现
    # 说明：除 approx_softplus 外，其余 "shifting" 仍按 MAC 计。
    # 新增：data movement 和 weight access 能耗
    # 对于FC/Linear/Conv层: T*s*(E_acc + E_move + E_weight)
    # =============================================================
    
    # =============================================================
    # in_proj (第一层Linear): D_IN -> D_OUT_PROJ, weight 32bit
    # 每来一次spike，需要weight access和move spike
    # weight access: T * s * (D_IN * D_OUT_PROJ * 32 bits)
    # spike move: T * s * (D_IN * 8 bits) 每个位置
    # =============================================================
    in_proj_weight_bits = B * L * T * s * (D_IN * D_OUT_PROJ * WEIGHT_BITS_FIRST_LINEAR)
    in_proj_move_bits = B * L  * (D_IN * ACTIVATION_BITS_in)
    add("in_proj (Linear, 32bit weight)", 
        mac=B*L*D_IN*D_OUT_PROJ, 
        acc=B*L*D_IN,
        move_bits=in_proj_move_bits,
        weight_bits=in_proj_weight_bits)

    # =============================================================
    # Depthwise conv1d (SpikeConv): kernel size 3
    # Conv层: weight是8bit
    # weight access: T * s * (kernel_size * 8 bits) 每个通道
    # spike move: T * s * (1 * 8 bits) 每个位置每个通道
    # =============================================================
    kernel_size = 3
    conv_weight_bits = B * DH * L * T * s * (kernel_size * WEIGHT_BITS_OTHER)
    conv_move_bits = B * DH * L * T * s * ACTIVATION_BITS
    add("Depthwise conv1d (SpikeConv, 8bit weight)", 
        mac=0, 
        acc=B*DH*L*kernel_size*T*s,
        move_bits=conv_move_bits,
        weight_bits=conv_weight_bits)

    # =============================================================
    # x_proj (SpikeFC): D_H -> (D_R+2D_S), weight 8bit
    # 对应流程图中 SSM 内部的 SpikeFC
    # =============================================================
    x_proj_weight_bits = B * L * T * s * (DH * DR2 * WEIGHT_BITS_OTHER)
    x_proj_move_bits = B * L * T * s * (DH * ACTIVATION_BITS)
    add("x_proj (SpikeFC D_H->D_R+2D_S, 8bit weight)", 
        mac=0, 
        acc=B*L*DH*DR2*T*s,
        move_bits=x_proj_move_bits,
        weight_bits=x_proj_weight_bits)

    # =============================================================
    # dt_proj (SpikeFC): D_R -> D_H, weight 8bit
    # 对应流程图中产生 delta 的 SpikeFC
    # =============================================================
    dt_proj_weight_bits = B * L * T * s * (DR * DH * WEIGHT_BITS_OTHER)
    dt_proj_move_bits = B * L * T * s * (DR * ACTIVATION_BITS)
    add("dt_proj (SpikeFC D_R->D_H, 8bit weight)", 
        mac=0, 
        acc=B*L*DH*DR*T*s,
        move_bits=dt_proj_move_bits,
        weight_bits=dt_proj_weight_bits)

    # =============================================================
    # approx_softplus (PTsoftplus): B*L*D_H*T*s
    # 这里改用 SHIFT 计能耗，无额外 data movement
    # =============================================================
    add("approx_softplus (PTsoftplus)", mac=0, acc=0, shift=B*L*DH*T*s)

    # =============================================================
    # Selective Scan 部分
    # A, D: 32bit, 只读取一次 (作为 weight)
    # B, C, delta, u: dataflow type, 无额外 data movement 能耗
    # h: 32bit, 在 t 时存储, 在 t-1 时读取 (data movement)
    # =============================================================
    
    # A (32bit) 读取一次: D_H * D_S * 32 bits
    #print(L,L,L)
    A_read_bits = L*B * DH * DS * STATE_BITS
    add("Scan: A weight read (32bit, once)", 
        mac=0, acc=0, 
        weight_bits=A_read_bits)
    
    # D (32bit) 读取一次: D_H * 32 bits
    D_read_bits = L*B * DH * STATE_BITS
    add("Scan: D weight read (32bit, once)", 
        mac=0, acc=0, 
        weight_bits=D_read_bits)

    # h state: 存储 L 次 (在每个 t), 读取 L-1 次 (在每个 t 需要 h_{t-1})
    # h size per step: B * D_H * D_S * 32 bits
    h_store_bits = L * B * DH * DS * STATE_BITS       # 存储 L 次
    h_read_bits = (L - 1) * B * DH * DS * STATE_BITS  # 读取 L-1 次 (第一个t不需要读h_{t-1})
    h_total_sram_bits = h_store_bits + h_read_bits
    add("Scan: h state read/write (32bit)", 
        mac=0, acc=0, 
        weight_bits=h_total_sram_bits)# 这里是读写SRAM

    # =============================================================
    # Selective Scan 计算部分 (Computing)
    # =============================================================
    
    # Einsum for DeltaA (SpikingEinsum): delta 和 A 做 einsum
    # delta 是 dataflow 无额外 move
    add("Scan: einsum DeltaA ACC", mac=0, acc=B*L*DH*DS*T*s)

    # Einsum for DeltaB / Bu (SpikingEinsum)
    add("Scan: DeltaB ACC #1", mac=0, acc=B*L*DH*DS*T*s)
    add("Scan: DeltaB ACC #2 (t-1)", mac=0, acc=B*L*DH*DS*(T-1)*s)
    add("Scan: DeltaB MAC #1", shift=B*L*DH*DS, acc=0)
    add("Scan: DeltaB ACC #3", mac=0, acc=B*L*DH*DS)
    add("Scan: DeltaB shift->MAC", shift=B*L*DH*DS, acc=0)
    add("Scan: DeltaB ACC #4", mac=0, acc=B*L*DH*DS*T*s)
    add("Scan: DeltaB ACC #5 (t-1)", mac=0, acc=B*L*DH*DS*(T-1)*s)

    # Loop (recurrence computation):
    # i=0: B*D_H*D_S shifting + B*D_H*D_S ACC
    add("Loop i=0: shift->MAC + ACC", shift=B*DH*DS, acc=B*DH*DS)

    # (L-1) * （B*D_H*D_S*T*s ACC + B*D_H*D_S*(T-1) ACC + B*D_H*D_S shifting + B*D_H*D_S ACC）
    add("Loop i=1..L-1 (per-step aggregated)",
        shift=(L-1)*(B*DH*DS),
        acc=(L-1)*(B*DH*DS*T*s + B*DH*DS*(T-1) + B*DH*DS))

    # reading y (SpikingEinsum for Ch_t):
    # L * (B*D_H*D_S*T*s ACC + B*D_H*T*(D_S-1) + B*D_H*(T-1) ACCs)
    add("Reading y over L (einsum Ch_t)",
        mac=0,
        acc=L*(B*DH*DS*T*s + B*DH*T*(DS-1) + B*DH*(T-1)))

    # Final terms (Du + output)
    # B*L*D_H*T*s ACCs + B*L*D_H*(T-1) ACCs + B*L*D_H shifting + B*L*D_H ACCs
    add("Final terms (Du + output)",
        shift=B*L*DH,
        acc=B*L*DH*T*s + B*L*DH*(T-1) + B*L*DH)

    # =============================================================
    # out_proj (QLinear): d_inner -> d_model, weight 8bit
    # 重复 3 次
    # d_inner = 64 (DH), d_model = 321 (D_IN)
    # =============================================================
    d_inner = DH      # 64
    d_model = D_IN    # 321
    
    out_proj_weight_bits =  B * L * T * s * (d_inner * d_model * WEIGHT_BITS_OTHER)
    out_proj_move_bits =   B * L * T * s * (d_inner * ACTIVATION_BITS)
    add("out_proj x3 (QLinear d_inner->d_model, 8bit weight)", 
        mac=0, 
        acc=  B * L * d_inner * d_model * T * s,
        move_bits=out_proj_move_bits,
        weight_bits=out_proj_weight_bits)

    # =============================================================
    # 汇总 (重复3次的部分)
    # =============================================================
    total_mac   = sum(r["MAC"] for r in rows)
    total_acc   = sum(r["ACC"] for r in rows)
    total_shift = sum(r["SHIFT"] for r in rows)
    total_move_bits = sum(r["MOVE_BITS"] for r in rows)
    total_weight_bits = sum(r["WEIGHT_BITS"] for r in rows)

    total_computing_energy_pJ = (
        total_mac   * MAC_pJ   +
        total_acc   * ACC_pJ   +
        total_shift * SHIFT_pJ
    )
    
    total_move_energy_pJ = total_move_bits * E_MOVE_PER_BIT
    total_weight_energy_pJ = total_weight_bits * E_WEIGHT_PER_BIT
    
    # 重复 3 次
    repeated_energy_pJ = 3 * (total_computing_energy_pJ + total_move_energy_pJ + total_weight_energy_pJ)

    # =============================================================
    # Final FC (分类头): 321 -> 3, weight 32bit, 只执行一次
    # input 是 32-bit
    # =============================================================
    final_fc_in = D_IN   # 321
    final_fc_out = 3
    FINAL_FC_WEIGHT_BITS = 32  # 32-bit weight
    FINAL_FC_INPUT_BITS = 32   # 32-bit input
    
    # MAC 操作 (非spike，所以是MAC)
    final_fc_mac = B * final_fc_in * final_fc_out
    final_fc_weight_bits = B * (final_fc_in * final_fc_out * FINAL_FC_WEIGHT_BITS)
    final_fc_move_bits = B * (final_fc_in * FINAL_FC_INPUT_BITS)
    
    final_fc_computing_energy = final_fc_mac * MAC_pJ
    final_fc_weight_energy = final_fc_weight_bits * E_WEIGHT_PER_BIT
    final_fc_move_energy = final_fc_move_bits * E_MOVE_PER_BIT
    final_fc_total_energy = final_fc_computing_energy + final_fc_weight_energy + final_fc_move_energy

    # 总能耗 = 重复3次的能耗 + final FC的能耗
    grand_total_energy_pJ = repeated_energy_pJ + final_fc_total_energy

    # 打印结果
    print("===== Parameters =====")
    for k, v in asdict(p).items():
        print(f"{k}: {v}")

    print("\n===== Per-Stage Breakdown (per layer, before x3) =====")
    for r in rows:
        e_mac   = r["MAC"]   * MAC_pJ
        e_acc   = r["ACC"]   * ACC_pJ
        e_shift = r["SHIFT"] * SHIFT_pJ
        e_move  = r["MOVE_BITS"] * E_MOVE_PER_BIT
        e_weight = r["WEIGHT_BITS"] * E_WEIGHT_PER_BIT
        e_computing = e_mac + e_acc + e_shift
        e_tot   = e_computing + e_move + e_weight
        print(f"{r['Stage']}:")
        print(f"  Computing: MAC={r['MAC']:.0f}, ACC={r['ACC']:.0f}, SHIFT={r['SHIFT']:.0f}")
        print(f"  Data Move: {r['MOVE_BITS']:.0f} bits, Weight Access: {r['WEIGHT_BITS']:.0f} bits")
        print(f"  Energy: E_computing={e_computing:.1f} pJ, E_move={e_move:.1f} pJ, E_weight={e_weight:.1f} pJ")
        print(f"  E_TOTAL={e_tot:.1f} pJ")
        print()

    print("===== TOTAL (Repeated x3) =====")
    print(f"TOTAL MAC       = {3*total_mac:.0f}")
    print(f"TOTAL ACC       = {3*total_acc:.0f}")
    print(f"TOTAL SHIFT     = {3*total_shift:.0f}")
    print(f"TOTAL MOVE_BITS = {3*total_move_bits:.0f} bits")
    print(f"TOTAL WEIGHT_BITS = {3*total_weight_bits:.0f} bits")
    print()
    print(f"Computing Energy (x3) = {3*total_computing_energy_pJ:.1f} pJ")
    print(f"Data Movement Energy (x3) = {3*total_move_energy_pJ:.1f} pJ")
    print(f"Weight Access Energy (x3) = {3*total_weight_energy_pJ:.1f} pJ")
    print(f"Subtotal (x3) = {repeated_energy_pJ:.1f} pJ")
    
    print("\n===== Final FC Layer (321 -> 3, 32bit) =====")
    print(f"MAC = {final_fc_mac:.0f}")
    print(f"Move bits = {final_fc_move_bits:.0f} bits")
    print(f"Weight bits = {final_fc_weight_bits:.0f} bits")
    print(f"Computing Energy = {final_fc_computing_energy:.1f} pJ")
    print(f"Data Movement Energy = {final_fc_move_energy:.1f} pJ")
    print(f"Weight Access Energy = {final_fc_weight_energy:.1f} pJ")
    print(f"Final FC Total = {final_fc_total_energy:.1f} pJ")
    
    print("\n===== GRAND TOTAL =====")
    print(f"TOTAL Energy = {grand_total_energy_pJ:.1f} pJ")

    return {
        "rows": rows,
        "TOTAL_MAC": 3*total_mac + final_fc_mac,
        "TOTAL_ACC": 3*total_acc,
        "TOTAL_SHIFT": 3*total_shift,
        "TOTAL_MOVE_BITS": 3*total_move_bits + final_fc_move_bits,
        "TOTAL_WEIGHT_BITS": 3*total_weight_bits + final_fc_weight_bits,
        "Computing_Energy_pJ": 3*total_computing_energy_pJ + final_fc_computing_energy,
        "Data_Movement_Energy_pJ": 3*total_move_energy_pJ + final_fc_move_energy,
        "Weight_Access_Energy_pJ": 3*total_weight_energy_pJ + final_fc_weight_energy,
        "TOTAL_Energy_pJ": grand_total_energy_pJ
    }

if __name__ == "__main__":
    # 可按需修改参数
    params = Params(
        B=1,
        L=168,
        D_H=64,
        D_S=32,
        T=3,
        s=0.268,
        D_R=21,
        D_IN=321,
        D_OUT_PROJ=128,
        # D_R_plus_2DS=None  # 留空自动用 D_R + 2*D_S
    )
    compute_counts(params)
