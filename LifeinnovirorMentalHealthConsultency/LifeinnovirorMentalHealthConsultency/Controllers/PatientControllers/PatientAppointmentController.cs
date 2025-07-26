using System;
using System.Data.Entity;
using System.Linq;
using System.Net;
using System.Threading.Tasks;
using System.Web.Http;
using LifeinnovirorMentalHealthConsultency.Context;
using LifeinnovirorMentalHealthConsultency.Context.Tables;
using LifeinnovirorMentalHealthConsultency.Functional_Class;
using LifeinnovirorMentalHealthConsultency.Models;

namespace LifeinnovirorMentalHealthConsultency.Controllers.PatientControllers
{
    public class PatientAppointmentController : ApiController
    {
        private readonly LifeinnovirorContext db;
        public PatientAppointmentController()
        {
            db = new LifeinnovirorContext();
        }


        [HttpPost]
        [Route("api/appointments/book")]
        public async Task<IHttpActionResult> BookAppointment([FromBody] AppointmentBookModel modelDTO)
        {
            try
            {
                ModelState.Remove("PatientId");

                if (!ModelState.IsValid)
                {
                    var errors = ModelState.Where(ms => ms.Value.Errors.Count > 0)
                                           .Select(ms => new
                                           {
                                               Field = ms.Key,
                                               Errors = ms.Value.Errors.Select(e => e.ErrorMessage).ToList()
                                           });

                    return Content(HttpStatusCode.BadRequest, new
                    {
                        success = false,
                        message = "Validation failed.",
                        errors = errors,
                        data = modelDTO
                    });
                }

                var model = new Appointment
                {
                    FullName = modelDTO.FullName,
                    Email = modelDTO.Email,
                    DoctorId = modelDTO.DoctorId,
                    SlotId = modelDTO.SlotId,
                    AppointmentTypeId = modelDTO.AppointmentTypeId,
                    MeetingMedium = modelDTO.MeetingMedium,
                    Notes = modelDTO.Notes,
                    BookedAt = DateTime.Now,
                    Status = "Booked" // default value
                };


                //check if correct doctor data came or not
                var doctor = await db.Doctors.FindAsync(model.DoctorId);
                if (doctor == null)
                {
                    return Content(HttpStatusCode.BadRequest, new
                    {
                        success = false,
                        message = "Missing doctor data."
                    });
                }

                // Check if the Slot is valid 
                var slot = await db.DoctorTimeSlots.FindAsync(model.SlotId);
                if (slot == null)
                {
                    return Content(HttpStatusCode.BadRequest, new
                    {
                        success = false,
                        message = "Invalid Slot Data."
                    });
                }

                //check if it is that particulat doctor timeslot or not
                if (slot.DoctorId != doctor.DoctorId)
                {
                    return Content(HttpStatusCode.BadRequest, new
                    {
                        success = false,
                        message = "This slot is not for that doctor!"
                    });
                }

                // check if it is in the past or not
                DateTime slotStartDateTime = slot.Date.Date + slot.StartTime;
                DateTime now = DateTime.Now;

                if (slotStartDateTime <= now)
                {
                    return Content(HttpStatusCode.BadRequest, new
                    {
                        success = false,
                        message = "Cannot book a slot in the past."
                    });
                }

                // check if slot not already booked
                if (slot.IsBooked)
                {
                    return Content(HttpStatusCode.BadRequest, new
                    {
                        success = false,
                        message = "This slot is already booked"
                    });
                }

                // check if appointmenttype exists or not
                var appointmentType = await db.AppointmentTypes.FindAsync(model.AppointmentTypeId);
                if (appointmentType == null)
                {
                    return Content(HttpStatusCode.BadRequest, new
                    {
                        success = false,
                        message = "Invalid Appointment Type"
                    });
                }


                // Check if Patient already exists 
                var existingPatient = await db.Patients
                    .FirstOrDefaultAsync(p => p.Email == model.Email);

                if (existingPatient != null)
                {
                    model.PatientId = existingPatient.PatientId;
                    //then check if this patient already has appointment in that day and that time

                    // first Get full DateTime range of selected slot
                    DateTime selectedStart = slot.Date.Date + slot.StartTime;
                    DateTime selectedEnd = slot.Date.Date + slot.EndTime;

                    var patientAppointments = await db.Appointments
                        .Include(a => a.Slot)
                        .Where(a => a.PatientId == model.PatientId && a.Status != "Cancelled")
                        .ToListAsync();

                    bool hasOverlap = patientAppointments.Any(a =>
                    {
                        DateTime existingStart = a.Slot.Date.Date + a.Slot.StartTime;
                        DateTime existingEnd = a.Slot.Date.Date + a.Slot.EndTime;

                        return selectedStart < existingEnd && selectedEnd > existingStart;
                    });


                    if (hasOverlap)
                    {
                        return Content(HttpStatusCode.Conflict, new
                        {
                            success = false,
                            message = "This time slot overlaps with your existing booked slot."
                        });
                    }

                }
                else // if patient not exit then create an account for him/her
                {
                    // New Patient: save and assign ID
                    var newPatient = new Patient
                    {
                        FullName = model.FullName,
                        Email = model.Email,
                        PasswordHash = CustomFunctions.GetSha256HashBase64(model.Email)
                    };
                    db.Patients.Add(newPatient);
                    await db.SaveChangesAsync();

                    model.PatientId = newPatient.PatientId;
                }


                // generate meeting link (not finalized)
                if (model.MeetingMedium?.Equals("Online", StringComparison.OrdinalIgnoreCase) == true)
                {
                    var meetingHelper = new TeamsMeetingHelper();
                    DateTime start = slot.Date.Date + slot.StartTime;
                    DateTime end = slot.Date.Date + slot.EndTime;

                    string meetingLink = await meetingHelper.CreateMeeting("Appointment", start, end);

                    if (meetingLink.StartsWith("ERROR"))
                    {
                        return Content(HttpStatusCode.InternalServerError, new
                        {
                            success = false,
                            message = "Unexpected error while creating meeting link",
                            error = meetingLink
                        });
                    }

                    model.MeetingLink = meetingLink;
                }

                // Set metadata
                model.BookedAt = DateTime.Now;
                // Save the Appointment
                db.Appointments.Add(model);
                // Mark slot as booked
                slot.IsBooked = true;

                // Create initial payment 
                var payment = new Payment
                {
                    AppointmentId = model.AppointmentId,
                    Amount = appointmentType.Cost,
                    Method = null,  // initially not paid
                    Status = "Pending",
                    TransactionId = null
                };
                db.Payments.Add(payment);


                // add notification for doctor
                var notification = new Notification
                {
                    RecipientType = "Doctor",
                    RecipientId = model.DoctorId,
                    Message = $"A new '{appointmentType.Name}' appointment has been booked " +
                              $"with you by patient {model.FullName} on " +
                              $"{slot.Date.Date} at {slot.StartTime}." +
                              (model.MeetingLink != null ? " Your online meeting link is: " + model.MeetingLink : ""),

                    SentAt = DateTime.Now,
                    Read = false
                };
                db.Notifications.Add(notification);

                // add notification for patient
                var notificationForPatient = new Notification
                {
                    RecipientType = "Patient",
                    RecipientId = model.PatientId,
                    Message = $"Your '{appointmentType.Name}' appointment with Dr. {doctor.FullName} " +
                              $"has been successfully booked for {slot.Date.Date} " +
                              $"at {slot.StartTime}." +
                              (model.MeetingLink != null ? " Your online meeting link is: " + model.MeetingLink : ""),

                    SentAt = DateTime.Now,
                    Read = false
                };
                db.Notifications.Add(notificationForPatient);

                // Log: 
                db.SystemLogs.Add(new SystemLog
                {
                    ActorType = "Patient",
                    ActorId = model.PatientId,
                    Action = "Booked an Appointment",
                    Details = $"Patient, Id='{model.PatientId}', Booked an appointment, Id={model.AppointmentId}.",
                    CreatedAt = DateTime.Now
                });
                await db.SaveChangesAsync();

                // sending necessary mail
                string patientAutoAccountCreationMail = "";
                if (existingPatient == null)
                {
                    patientAutoAccountCreationMail = EmailManagement.PatientAutoAccountCreationMail(model.FullName, model.Email);
                }
                string doctorNewAppointmentNotificationMail = EmailManagement.DoctorNewAppointmentNotificationMail(
                                            doctor.FullName,          // Doctor's Name
                                            doctor.Email,             // Doctor's Email
                                            model.FullName,           // Patient's Name
                                            slot.Date,                // Appointment Date
                                            slot.StartTime,           // Appointment Start Time
                                            model.MeetingLink != null, // IsOnline
                                            model.MeetingLink         // Meeting Link (optional)
                                        );
                string patientAppointmentBookingMail = EmailManagement.PatientAppointmentBookingMail(
                                model.FullName,          // Doctor's Name
                                model.Email,             // Doctor's Email
                                doctor.FullName,           // Patient's Name
                                slot.Date,                // Appointment Date
                                slot.StartTime,           // Appointment Start Time
                                model.MeetingLink != null, // IsOnline
                                model.MeetingLink         // Meeting Link (optional)
                            );

                return Ok(new
                {
                    success = true,
                    message = "Appointment added successfully.",
                    mailStatusReport = new
                    {
                        patientAutoAccountCreationMail = patientAutoAccountCreationMail,
                        doctorNewAppointmentNotificationMail = doctorNewAppointmentNotificationMail,
                        patientAppointmentBookingMail = patientAppointmentBookingMail
                    }
                });
            }
            catch (Exception ex)
            {
                return Content(HttpStatusCode.InternalServerError, new
                {
                    success = false,
                    message = "An error occurred while booking an appointment.",
                    error = ex.Message
                });
            }
        }
    }
}
