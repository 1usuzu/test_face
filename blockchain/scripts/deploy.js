const hre = require("hardhat");

async function main() {
  console.log("Deploying DeepfakeVerification contract...\n");

  // Get deployer account
  const [deployer] = await hre.ethers.getSigners();
  console.log("Deploying with account:", deployer.address);
  
  const balance = await deployer.provider.getBalance(deployer.address);
  console.log("Account balance:", hre.ethers.formatEther(balance), "ETH\n");

  // Deploy contract (no constructor args - oracle signer set separately)
  const DeepfakeVerification = await hre.ethers.getContractFactory("DeepfakeVerification");
  const contract = await DeepfakeVerification.deploy();

  await contract.waitForDeployment();
  
  const contractAddress = await contract.getAddress();
  console.log("DeepfakeVerification deployed to:", contractAddress);
  console.log("\nSave this address for frontend integration!");
  
  // Verify deployment
  const owner = await contract.owner();
  console.log("Contract owner:", owner);
  
  const stats = await contract.getStats();
  console.log("Initial stats - DIDs:", stats[0].toString(), ", Verifications:", stats[1].toString());
  
  // Save deployment info
  const fs = require("fs");
  const path = require("path");
  
  const deploymentInfo = {
    network: hre.network.name,
    contractAddress: contractAddress,
    deployer: deployer.address,
    deployedAt: new Date().toISOString()
  };
  
  fs.writeFileSync(
    "./deployment.json", 
    JSON.stringify(deploymentInfo, null, 2)
  );
  console.log("\nDeployment info saved to deployment.json");

  // Auto-update frontend .env
  const envPath = path.join(__dirname, "../../frontend/.env");
  
  // Xác định Chain ID dựa trên network
  const chainIds = {
    localhost: 31337,
    hardhat: 31337,
    sepolia: 11155111,
    amoy: 80002
  };
  const chainId = chainIds[hre.network.name] || 31337;
  
  const networkNames = {
    localhost: "Hardhat Local",
    hardhat: "Hardhat Local",
    sepolia: "Sepolia",
    amoy: "Polygon Amoy"
  };
  const networkName = networkNames[hre.network.name] || "Unknown";
  
  if (fs.existsSync(envPath)) {
    let envContent = fs.readFileSync(envPath, "utf8");
    // Replace contract address
    envContent = envContent.replace(
      /VITE_CONTRACT_ADDRESS=.*/,
      `VITE_CONTRACT_ADDRESS=${contractAddress}`
    );
    // Replace chain ID
    envContent = envContent.replace(
      /VITE_CHAIN_ID=.*/,
      `VITE_CHAIN_ID=${chainId}`
    );
    // Replace network name
    if (envContent.includes("VITE_NETWORK_NAME=")) {
      envContent = envContent.replace(
        /VITE_NETWORK_NAME=.*/,
        `VITE_NETWORK_NAME=${networkName}`
      );
    }
    fs.writeFileSync(envPath, envContent);
    console.log(`Frontend .env updated! (Contract: ${contractAddress}, Chain: ${chainId})`);
  } else {
    // Create new .env file
    const newEnv = `VITE_API_URL=http://localhost:8000
VITE_CONTRACT_ADDRESS=${contractAddress}
VITE_CHAIN_ID=${chainId}
VITE_NETWORK_NAME=${networkName}
`;
    fs.writeFileSync(envPath, newEnv);
    console.log("Frontend .env created with contract address!");
  }
}

main()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error("❌ Error:", error);
    process.exit(1);
  });
