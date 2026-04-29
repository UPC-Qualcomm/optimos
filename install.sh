#!/bin/bash

set -euo pipefail

# Resolve the directory containing this script — that is the OPTIMOS root.
OPTIMOS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ASTRA_SIM="${OPTIMOS_ROOT}/astra-sim"

if [ ! -d "${ASTRA_SIM}" ]; then
    echo "Error: astra-sim submodule not found at ${ASTRA_SIM}" >&2
    echo "Run: git submodule update --init --recursive" >&2
    exit 1
fi

# Phase 1: Ensure pyenv + Python 3.11.11 are available and use that exact
# interpreter to create the optimosenv virtual environment.
PYENV_ROOT="${HOME}/.pyenv"
if [ ! -x "${PYENV_ROOT}/bin/pyenv" ]; then
    if command -v curl >/dev/null 2>&1; then
        curl -fsSL https://pyenv.run | bash
    elif command -v wget >/dev/null 2>&1; then
        wget -qO- https://pyenv.run | bash
    else
        echo "Error: need curl or wget to install pyenv" >&2
        exit 1
    fi
fi

export PYENV_ROOT
export PATH="${PYENV_ROOT}/bin:${PATH}"
eval "$(pyenv init - bash)"

if ! pyenv versions --bare | grep -qx "3.11.11"; then
    pyenv install 3.11.11
fi

PY311="${PYENV_ROOT}/versions/3.11.11/bin/python3.11"

# Create the virtualenv at the optimos root (not inside astra-sim).
"${PY311}" -m venv "${OPTIMOS_ROOT}/optimosenv"

source "${OPTIMOS_ROOT}/optimosenv/bin/activate"

pip3 install --upgrade pip

pip3 install protobuf==5.29.0

pip3 install graphviz pydot sympy tqdm seaborn matplotlib

pip3 install scikit-learn altair scipy umap-learn xgboost intervaltree ipykernel

cd "${ASTRA_SIM}"

git submodule update --init --recursive

ASTRA_SIM_BIN_AWARE="${ASTRA_SIM}/build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Aware"
ASTRA_SIM_BIN_UNAWARE="${ASTRA_SIM}/build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Unaware"
ASTRA_SIM_PYTHON="${OPTIMOS_ROOT}/optimosenv/bin/python"

upsert_bashrc_export() {
    local key="$1"
    local value="$2"
    local bashrc_file="${HOME}/.bashrc"

    if grep -q "^export ${key}=" "${bashrc_file}"; then
        sed -i "s|^export ${key}=.*|export ${key}=${value}|" "${bashrc_file}"
    else
        echo "export ${key}=${value}" >> "${bashrc_file}"
    fi
}

upsert_bashrc_export "PYENV_ROOT" '"$HOME/.pyenv"'
upsert_bashrc_export "OPTIMOS_ROOT" "${OPTIMOS_ROOT}"
upsert_bashrc_export "ASTRA_SIM_BIN_AWARE" "${ASTRA_SIM_BIN_AWARE}"
upsert_bashrc_export "ASTRA_SIM_BIN_UNAWARE" "${ASTRA_SIM_BIN_UNAWARE}"
upsert_bashrc_export "ASTRA_SIM_ROOT" "${ASTRA_SIM}"
upsert_bashrc_export "ASTRA_SIM_PYTHON" "${ASTRA_SIM_PYTHON}"

if ! grep -q 'pyenv init - bash' "${HOME}/.bashrc"; then
    {
        echo '[[ -d "$PYENV_ROOT/bin" ]] && export PATH="$PYENV_ROOT/bin:$PATH"'
        echo 'eval "$(pyenv init - bash)"'
    } >> "${HOME}/.bashrc"
fi

# Chakra's ETFeederNode::get_chakra_node() is kept private upstream, but
# AstraSim's workload code needs to call it during the build; patch it here so
# the install remains self-contained and does not depend on a separate repo fix.
python3 - <<'PY'
from pathlib import Path

header = Path('extern/graph_frontend/chakra/src/feeder_v3/et_feeder_node.h')
lines = header.read_text().splitlines(keepends=True)

target_idx = -1
for i, line in enumerate(lines):
    if 'get_chakra_node() const;' in line and 'shared_ptr' in line:
        target_idx = i
        break

if target_idx == -1:
    raise SystemExit('Failed to find ETFeederNode::get_chakra_node() declaration')

# If the nearest previous non-empty line is already "public:", nothing to do.
prev_non_empty = target_idx - 1
while prev_non_empty >= 0 and lines[prev_non_empty].strip() == '':
    prev_non_empty -= 1

if prev_non_empty >= 0 and lines[prev_non_empty].strip() == 'public:':
    raise SystemExit(0)

indent = lines[target_idx][: len(lines[target_idx]) - len(lines[target_idx].lstrip())]
lines.insert(target_idx, f'{indent}public:\n')
header.write_text(''.join(lines))
PY

./build/astra_analytical/build.sh

./build/astra_ns3/build.sh -c

cd ./extern/graph_frontend/chakra/

pip3 install .

cd "${ASTRA_SIM}"
pip3 install -r "${OPTIMOS_ROOT}/requirements.txt"

source ~/.bashrc
