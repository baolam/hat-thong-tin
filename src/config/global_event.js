const events = require("events");
const EventEmiter = new events.EventEmitter();
EventEmiter.setMaxListeners(100);

const TYPE = {
    GET_DEFAULT_USER : "GET_DEFAULT_USER"
}

module.exports = {
    GlobalEvents : EventEmiter,
    TYPE
};