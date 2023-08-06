const express = require("express");
const Router = express.Router();

const user = require("../models/UserModel");
/// Kiểm tra đã login hay chưa
// Router.get("/login", (req, res) => {
//     res.send(req.signedCookies.user_id);
// });

Router.post("/login", (req, res) => {
    let { mail, password } = req.body;
    (async () => {
        try {
            let [state, id] = user.logIn(mail, password);
            if (! state)
                throw new Error("Wrong password");
            else {
                res.cookie("user_id", id, { 
                    signed : true,
                    expires : new Date(Date.now() + 12 * 60 * 60 * 1000) 
                });
                res.json({ msg : "" });
            }
        } catch(err)
        {
            res.json({ msg : err, code : false });
        }
    })();
});

const upload = require("../middleware/UserMulter");
Router.post("/register", upload.single("avatar"), (req, res) => {
    let { name, mail, password } = req.body;
    let avatar = req.file.fieldname;
    user.onCreate(name, mail, password, avatar)
        .then(() => res.json({ msg : "" }))
        .catch(err => res.json({ msg : err }));
});

module.exports = Router;