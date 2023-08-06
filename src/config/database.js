const sequelize = require("sequelize");
const { info } = require("./logger");

const path = require("path");
const Database = new sequelize.Sequelize({
    // username : process.env.DB_USER,
    // password : process.env.DB_PASS,
    // database : process.env.DATABASE,
    // host : process.env.DB_HOST,
    dialect : "sqlite",
    storage : path.join(__dirname, "../../", process.env.DATABASE),
    port : 5432,
    sync : true,
    logging : (sql, _timing) => {
        info(sql);
    }
});

module.exports = Database;