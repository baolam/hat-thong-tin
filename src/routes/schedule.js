const express = require("express");
const Router = express.Router();

const schedule = require("../models/ScheduleModels").schedule;
const { info, error } = require("../config/logger.js");

Router.get("/", (_req, res) => {
    schedule.onGetList()
        .then(vl => res.json(vl))
        .catch(er => {
            res.json([]);
            log_err(er);
        });
});

Router.post("/", (req, res) => {
    let { 
        name, 
        message, 
        cron_string 
    } = req.body;
    schedule.onCreate(name, cron_string, message)
        .then(msg => info(msg))
        .catch(err => error(err));
    res.send("OK!");
});

Router.put("/", (req, res) => {
    let { name, wanna_update } = req.body;
})

Router.delete("/", (req, res) => {
    let name = req.body.name;
    schedule.onDeleteBaseName(name);
    res.send("OK!");
});

/// TEST
// Thử tạo một plan để test
Router.get("/test/create", (req, res) => {
    let { 
        name,
        message,
        cron_string
    } = req.query;
    schedule.onCreate(name, cron_string, message)
        .then(msg => console.log(msg))
        .catch(err => console.log(err));
    res.send("OK!");
});

Router.get("/test/delete", (req, res) => {
    let name = req.query.name;
    schedule.onDeleteBaseName(name);
    res.send("OK!");
});

module.exports = Router;