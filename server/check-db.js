const sqlite3 = require('sqlite3').verbose();
const path = require('path');
const dbPath = path.join(__dirname, '..', 'satyaai.db');

console.log(`🔍 Checking database at: ${dbPath}`);

// Check if database file exists
const fs = require('fs');
if (!fs.existsSync(dbPath)) {
  console.log('❌ Database file not found');
  process.exit(1);
}

console.log('✅ Database file exists');

// Try to connect to the database
const db = new sqlite3.Database(dbPath, sqlite3.OPEN_READONLY, (err) => {
  if (err) {
    console.error('❌ Could not connect to database', err.message);
    return;
  }
  
  console.log('✅ Successfully connected to the SQLite database');
  
  // List all tables
  db.all("SELECT name FROM sqlite_master WHERE type='table'", [], (err, tables) => {
    if (err) {
      console.error('❌ Error fetching tables', err.message);
      return;
    }
    
    console.log('\n📊 Database Tables:');
    console.log(tables.length > 0 
      ? tables.map(t => `- ${t.name}`).join('\n')
      : 'No tables found');
    
    // Close the database connection
    db.close();
  });
});
