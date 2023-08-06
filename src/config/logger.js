const { createLogger, format } = require("winston");
const { combine, timestamp, printf, errors, json } = format;
const { HttpLogger, DailyLogger } = require("./#config_logger");

const print_format = printf((info) => {
    let s = `${info.level} : [${info.timestamp}]~[] : ${info.message}`;
    if (info.level != "error")
        return s
    s = `${s} ~ ${info.stack}`;
    return s;
});

// addColors(colors = {
//     error: 'red',
//     warn: 'yellow',
//     info: 'green',
//     http: 'magenta',
//     debug: 'white',
// });

const logger = createLogger({
    format : combine(
        timestamp(),
        // colorize({ all : true }),
        errors({ stack : true }),
        json(),
        print_format,
        // prettyPrint()
    ),
    transports : [
        DailyLogger, HttpLogger
    ]
});

function info(msg)
{
    logger.info({
        message : msg
    });
}

function error(msg)
{
    logger.error({
        message : msg
    });
}

process.on("uncaughtException", (err) => {
    logger.log("error", err);
});

module.exports = {
    logger,
    info, error
};