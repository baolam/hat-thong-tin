const shortid = require("shortid");
const multer = require("multer");

const storage = multer.diskStorage({
    destination : (req, file, cb) => {
        cb(null, process.env.USER_FOLDER)
    },
    filename : (req, file, cb) => {
        cb(null, `${shortid.generate()}-${file.filename}-(${Date.now()}).png`);
    }
});

const upload = multer({ storage });
module.exports = upload;