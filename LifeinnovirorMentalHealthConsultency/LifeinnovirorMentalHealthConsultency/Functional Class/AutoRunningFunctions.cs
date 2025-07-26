using System;
using System.Collections.Generic;
using System.Data.Entity;
using System.Linq;
using System.Net;
using System.Threading.Tasks;
using System.Web;
using LifeinnovirorMentalHealthConsultency.Context;
using LifeinnovirorMentalHealthConsultency.Context.Tables;

namespace LifeinnovirorMentalHealthConsultency.Functional_Class
{
    public class AutoRunningFunctions
    {
        private readonly LifeinnovirorContext db;
        public AutoRunningFunctions()
        {
            db = new LifeinnovirorContext();
        }


        public async Task CancelUnpaidExpiredAppointments()
        {
            try
            {
                // Step 1: Get all appointments with pending payments
                var pendingPayments = await db.Payments
                    .Where(p => p.Status == "Pending")
                    .ToListAsync();

                foreach (var payment in pendingPayments)
                {
                    var appointment = await db.Appointments
                        .Include(a => a.Slot)
                        .Include(a => a.Doctor)
                        .FirstOrDefaultAsync(a => a.AppointmentId == payment.AppointmentId);

                    if (appointment == null || appointment.Status == "Cancelled")
                        continue;

                    var slot = appointment.Slot;
                    var doctor = appointment.Doctor;

                    if (slot == null || doctor == null)
                        continue;

                    // Step 2: Compute payment deadline dynamically
                    DateTime appointmentDateTime = slot.Date.Date + slot.StartTime;
                    DateTime paymentDeadline = appointmentDateTime.AddDays(-doctor.minimumCancelTime);

                    if (DateTime.Now > paymentDeadline)
                    {
                        // Step 3: Cancel the appointment
                        appointment.Status = "Cancelled";
                        appointment.CancellationReason = "Payment deadline exceeded.";
                        appointment.CancelledAt = DateTime.Now;

                        // Step 4: Mark payment as failed
                        payment.Status = "Failed";

                        // Step 5: Mark slot as available again
                        slot.IsBooked = false;

                        // Step 6: Notify the patient 
                        var patientNotification = new Notification
                        {
                            RecipientType = "Patient",
                            RecipientId = appointment.PatientId,
                            Message = $"Your appointment on {slot.Date} at {slot.StartTime} was cancelled due to non-payment.",
                            SentAt = DateTime.Now,
                            Read = false
                        };
                        db.Notifications.Add(patientNotification);

                        // Step 7: Notify the doctor 
                        var dcotorNotification = new Notification
                        {
                            RecipientType = "Doctor",
                            RecipientId = appointment.DoctorId,
                            Message = $"Your appointment on {slot.Date.Date} at {slot.StartTime} was cancelled due to non-payment.",
                            SentAt = DateTime.Now,
                            Read = false
                        };
                        db.Notifications.Add(dcotorNotification);

                        // Step 8: Log the cancellation
                        db.SystemLogs.Add(new SystemLog
                        {
                            ActorType = "System",
                            ActorId = 0,
                            Action = "Auto Cancelled Appointment",
                            Details = $"Appointment ID {appointment.AppointmentId} cancelled due to missed payment deadline.",
                            CreatedAt = DateTime.Now
                        });
                    }
                }

                await db.SaveChangesAsync();
            }
            catch (Exception ex)
            {
                // Log or notify error
                try
                {
                    db.SystemLogs.Add(new SystemLog
                    {
                        ActorType = "System",
                        ActorId = 0,
                        Action = "Error in PaymentDeadlineChecker",
                        Details = $"Exception: {ex.Message}",
                        CreatedAt = DateTime.Now
                    });

                    // Notify the admin 
                    var adminNotification = new Notification
                    {
                        RecipientType = "Admin",
                        RecipientId = 0,  //everyone
                        Message = $"An Error occured in auto-appointment-cancellation method. Error: "+ex.Message,
                        SentAt = DateTime.Now,
                        Read = false
                    };
                    db.Notifications.Add(adminNotification);

                    await db.SaveChangesAsync();
                }
                catch (Exception) { }
            }
        }
    }


}