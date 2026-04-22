const hre = require("hardhat");
const fs = require("fs");
const path = require("path");

const VALID_ONLY_MODES = new Set(["all", "deepfake", "voting"]);

function parseArgs(argv) {
  const args = {
    only: process.env.DEPLOY_ONLY || "all",
    skipFrontendEnv: process.env.SKIP_FRONTEND_ENV === "true",
  };
  for (let i = 0; i < argv.length; i++) {
    const token = argv[i];
    if (token === "--only") {
      args.only = argv[i + 1] || "all";
      i++;
    } else if (token === "--skip-frontend-env") {
      args.skipFrontendEnv = true;
    }
  }
  if (!VALID_ONLY_MODES.has(args.only)) {
    throw new Error(`Invalid --only value "${args.only}". Use one of: all | deepfake | voting`);
  }
  return args;
}

function getNetworkConfig(networkName) {
  const chainIds = { localhost: 31337, hardhat: 31337, sepolia: 11155111, amoy: 80002 };
  const networkNames = { localhost: "Hardhat Local", hardhat: "Hardhat Local", sepolia: "Sepolia", amoy: "Polygon Amoy" };
  const rpcUrls = {
    localhost: "http://127.0.0.1:8545",
    hardhat: "http://127.0.0.1:8545",
    sepolia: process.env.SEPOLIA_RPC || "https://rpc.sepolia.org",
    amoy: process.env.POLYGON_AMOY_RPC || "https://rpc-amoy.polygon.technology/",
  };
  return {
    chainId: chainIds[networkName] || 31337,
    networkName: networkNames[networkName] || "Unknown",
    rpcUrl: rpcUrls[networkName] || "http://127.0.0.1:8545",
  };
}

async function estimateDeployCostWei(factory, deployerAddress) {
  const deployTx = await factory.getDeployTransaction();
  const provider = hre.ethers.provider;
  const feeData = await provider.getFeeData();
  const gasPrice = getGasPriceForEstimate(feeData);
  if (!gasPrice) {
    throw new Error("Could not determine gas price from provider.");
  }
  const gasLimit = await provider.estimateGas({ ...deployTx, from: deployerAddress });
  return gasLimit * gasPrice;
}

function gweiToWeiBigInt(value) {
  if (value === undefined || value === null || value === "") return null;
  return hre.ethers.parseUnits(String(value), "gwei");
}

function getTxFeeOverrides() {
  const gasPrice = gweiToWeiBigInt(process.env.TX_GAS_PRICE_GWEI);
  const maxFeePerGas = gweiToWeiBigInt(process.env.TX_MAX_FEE_GWEI);
  const maxPriorityFeePerGas = gweiToWeiBigInt(process.env.TX_MAX_PRIORITY_FEE_GWEI);

  if (gasPrice) {
    return { gasPrice };
  }
  if (maxFeePerGas || maxPriorityFeePerGas) {
    return {
      ...(maxFeePerGas ? { maxFeePerGas } : {}),
      ...(maxPriorityFeePerGas ? { maxPriorityFeePerGas } : {}),
    };
  }
  return {};
}

function getGasPriceForEstimate(feeData) {
  const overrides = getTxFeeOverrides();
  if (overrides.gasPrice) return overrides.gasPrice;
  if (overrides.maxFeePerGas) return overrides.maxFeePerGas;
  return feeData.gasPrice ?? feeData.maxFeePerGas;
}

function upsertEnv(content, key, value) {
  const line = `${key}=${value}`;
  const regex = new RegExp(`^${key}=.*$`, "m");
  if (regex.test(content)) {
    return content.replace(regex, line);
  }
  const suffix = content.endsWith("\n") ? "" : "\n";
  return `${content}${suffix}${line}\n`;
}

function updateFrontendEnv({ deepfakeAddress, votingFactoryAddress, chainId, networkName, rpcUrl }) {
  const envPath = path.join(__dirname, "../../frontend/.env");
  const forceUpdateFrontendEnv = process.env.FORCE_UPDATE_FRONTEND_ENV === "true";
  const hasEnv = fs.existsSync(envPath);

  let content = hasEnv ? fs.readFileSync(envPath, "utf8") : "VITE_API_URL=http://localhost:8000\n";
  const currentChainMatch = content.match(/VITE_CHAIN_ID=(\d+)/);
  const currentChainId = currentChainMatch ? Number(currentChainMatch[1]) : null;

  if (!forceUpdateFrontendEnv && currentChainId && currentChainId !== chainId) {
    console.log(`Skipped frontend .env update: current VITE_CHAIN_ID=${currentChainId}, deploy network chainId=${chainId}.`);
    console.log("Set FORCE_UPDATE_FRONTEND_ENV=true to override.");
    return;
  }

  if (deepfakeAddress) content = upsertEnv(content, "VITE_CONTRACT_ADDRESS", deepfakeAddress);
  if (votingFactoryAddress) content = upsertEnv(content, "VITE_VOTING_FACTORY_ADDRESS", votingFactoryAddress);
  content = upsertEnv(content, "VITE_CHAIN_ID", String(chainId));
  content = upsertEnv(content, "VITE_NETWORK_NAME", networkName);
  content = upsertEnv(content, "VITE_RPC_URL", rpcUrl);
  fs.writeFileSync(envPath, content);

  console.log(`Frontend .env ${hasEnv ? "updated" : "created"} successfully.`);
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  console.log(`Deploy mode: ${args.only}\n`);

  const signers = await hre.ethers.getSigners();
  if (!signers.length) {
    throw new Error("No deployer signer found. Set PRIVATE_KEY in blockchain/.env and rerun.");
  }
  const [deployer] = signers;
  const provider = hre.ethers.provider;
  const startBalance = await provider.getBalance(deployer.address);

  console.log("Deploying with account:", deployer.address);
  console.log("Account balance:", hre.ethers.formatEther(startBalance), "ETH\n");

  const shouldDeployDeepfake = args.only === "all" || args.only === "deepfake";
  const shouldDeployVoting = args.only === "all" || args.only === "voting";
  const feeOverrides = getTxFeeOverrides();
  if (Object.keys(feeOverrides).length) {
    console.log("Using tx fee overrides from env:", feeOverrides, "\n");
  }

  let estimatedTotalCost = 0n;
  let deepfakeFactory = null;
  let votingFactoryFactory = null;

  if (shouldDeployDeepfake) {
    deepfakeFactory = await hre.ethers.getContractFactory("DeepfakeVerification");
    estimatedTotalCost += await estimateDeployCostWei(deepfakeFactory, deployer.address);
  }
  if (shouldDeployVoting) {
    votingFactoryFactory = await hre.ethers.getContractFactory("VotingFactory");
    estimatedTotalCost += await estimateDeployCostWei(votingFactoryFactory, deployer.address);
  }

  const bufferPercent = Number(process.env.DEPLOY_BUFFER_PERCENT || 20);
  const normalizedBufferPercent = Number.isFinite(bufferPercent) && bufferPercent >= 0 ? bufferPercent : 20;
  const requiredWithBuffer = (estimatedTotalCost * BigInt(100 + Math.floor(normalizedBufferPercent))) / 100n;
  console.log("Estimated deploy cost:", hre.ethers.formatEther(estimatedTotalCost), "ETH");
  console.log(`Required with ${Math.floor(normalizedBufferPercent)}% buffer:`, hre.ethers.formatEther(requiredWithBuffer), "ETH\n");

  if (startBalance < requiredWithBuffer) {
    throw new Error(
      `Insufficient funds for selected deploy mode. Need ~${hre.ethers.formatEther(requiredWithBuffer)} ETH with buffer, have ${hre.ethers.formatEther(startBalance)} ETH.`
    );
  }

  let deepfakeAddress = null;
  let votingFactoryAddress = null;

  if (shouldDeployDeepfake) {
    console.log("Deploying DeepfakeVerification contract...");
    const contract = await deepfakeFactory.deploy(feeOverrides);
    await contract.waitForDeployment();
    deepfakeAddress = await contract.getAddress();
    console.log("DeepfakeVerification deployed to:", deepfakeAddress);

    const owner = await contract.owner();
    const stats = await contract.getStats();
    console.log("Contract owner:", owner);
    console.log("Initial stats - DIDs:", stats[0].toString(), ", Verifications:", stats[1].toString(), "\n");
  }

  if (shouldDeployVoting) {
    console.log("Deploying VotingFactory...");
    const votingFactory = await votingFactoryFactory.deploy(feeOverrides);
    await votingFactory.waitForDeployment();
    votingFactoryAddress = await votingFactory.getAddress();
    console.log("VotingFactory deployed to:", votingFactoryAddress, "\n");
  }

  const net = getNetworkConfig(hre.network.name);
  const deploymentInfo = {
    network: hre.network.name,
    chainId: net.chainId,
    deepfakeContractAddress: deepfakeAddress,
    votingFactoryAddress,
    deployer: deployer.address,
    deployedAt: new Date().toISOString(),
  };

  const deploymentFileName = `deployment.${hre.network.name}.json`;
  fs.writeFileSync(path.join(__dirname, "..", deploymentFileName), JSON.stringify(deploymentInfo, null, 2));
  console.log(`Deployment info saved to ${deploymentFileName}`);

  console.log("\nBackend env hint:");
  if (deepfakeAddress) console.log(`  CONTRACT_ADDRESS=${deepfakeAddress}`);
  console.log(`  RPC_URL=${net.rpcUrl}`);
  console.log(`  CHAIN_ID=${net.chainId}`);
  if (votingFactoryAddress) console.log(`  VOTING_FACTORY_ADDRESS=${votingFactoryAddress}`);

  if (!args.skipFrontendEnv) {
    updateFrontendEnv({
      deepfakeAddress,
      votingFactoryAddress,
      chainId: net.chainId,
      networkName: net.networkName,
      rpcUrl: net.rpcUrl,
    });
  } else {
    console.log("Skipped frontend .env update (--skip-frontend-env).");
  }
}

main()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error("❌ Error:", error);
    process.exit(1);
  });
