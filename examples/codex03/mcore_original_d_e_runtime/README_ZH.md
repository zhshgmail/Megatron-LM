# 原 WikiText D100 → 自有 D50 → E100 的运行环境接入

本目录只补齐原 runner 的依赖路径合同，保留原 D 的87项和 E的88项训练 argv；不改模型、优化器、TE、Adaptor、严格比较器及输入。D/E完整模型运行尚未发生。

选定 MCore source533cf（algorithm39ea）；`pretrain_gpt.py` 已正常导入原 Adaptor f707。实际 CP312/aarch64 的正常 GPTDataset 消费曾因 training eager import 缺 pydantic 失败；仅挂载下面四个包及各自 dist-info 后，原数据、helper、批次与游标消费通过。该数据通过不证明 TE 模型 ABI 或训练通过。

Root 在真实目标准入后，将原 source、TE site、Adaptor、八件输入及原 observer 只读绑定；显式设置原 RUN_D/RUN_E 所列变量，新增 `RUNTIME_DEPENDENCY_ROOT` 指向仅含 `pydantic`、`pydantic_core`、`annotated_types`、`typing_inspection` 及其 dist-info 的只读挂载。`typing_extensions` 使用原已验证基础环境4.16.0。不要把完整已有 user-site 加入路径，它含旧 MCore/TE。

已验证 helper290904B 的 SHA256 为271ab78945aaef25c861425de84c5da713dd9f1afa9f1184f9870d231e29d08e。必须把真实 CP312/aarch64 helper只读挂载到选定 source 的 `megatron/core/datasets/helpers_cpp.cpython-312-aarch64-linux-gnu.so`；源目录的零字节占位文件不是二进制。无需重新编译或安装。

D 的 load 必须是新的空目录，D 独立生成自身50与100；E使用自身 D50 的只读selector，保留Dlatest100。输出、cache、端口由当前目标准入绑定，禁止借旧 checkpoint50或旧K0/K1 PASS。

在本地 CP311/无 TorchNPU 环境调用原 TE layer-spec factory，正常 MCore导入后因没有TE而在TESpecProvider处失败；这只界定本地缺少该provider，不能推出目标TE不兼容。旧TE47精确Git身份仍UNKNOWN，TE93备份snapshot不补足原pin；当前需在目标正常原模型入口核实际TE/Adaptor/DSO身份与ABI，随后才有资格执行原D/E。不能用CPU替代、mock或修改backend越过它。
